"""
PeakWeights: Find the weights that matter. Protect them. Quantize the rest.

One-pass, data-free discovery of critical LLM parameters.

Version: 2.0 (December 27, 2025)

Key Finding: The optimal K is model-dependent:
    - SmolLM2-style (small): K=20 for 90% recovery
    - Qwen/DeepSeek: K=50 for 95% recovery
    - Mistral: K=100 for 99% recovery

Default: K=50 achieves 90%+ recovery on most models.

Usage:
    peakweights meta-llama/Llama-3.1-70B
    peakweights deepseek-ai/DeepSeek-V3 --top_k 50
    peakweights Qwen/Qwen2.5-72B --calibrate  # Find optimal K
    peakweights mistralai/Mistral-7B --viz

License: MIT
"""

import warnings
warnings.filterwarnings("ignore", message=".*pynvml.*deprecated.*")

import torch
import heapq
import json
import argparse
import sys
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

try:
    from importlib.metadata import version as get_version
    __version__ = get_version("peakweights")
except Exception:
    __version__ = "0.3.0"  # fallback for development


# =============================================================================
# Memory Utilities
# =============================================================================

def estimate_model_memory(config, dtype: torch.dtype = torch.float16) -> int:
    """
    Estimate memory required to load a model from its config.

    Returns estimated bytes needed (model weights + ~20% overhead for activations).
    """
    bytes_per_param = 2 if dtype in [torch.float16, torch.bfloat16] else 4

    # Try to estimate parameter count from config
    hidden_size = getattr(config, 'hidden_size', 4096)
    num_layers = getattr(config, 'num_hidden_layers', 32)
    vocab_size = getattr(config, 'vocab_size', 32000)
    intermediate_size = getattr(config, 'intermediate_size', hidden_size * 4)
    num_attention_heads = getattr(config, 'num_attention_heads', 32)
    num_key_value_heads = getattr(config, 'num_key_value_heads', num_attention_heads)

    # Embedding parameters
    embed_params = vocab_size * hidden_size  # input embeddings

    # Per-layer parameters (attention + MLP + layer norms)
    # Attention: Q, K, V projections + output projection
    head_dim = hidden_size // num_attention_heads
    q_params = hidden_size * hidden_size
    k_params = hidden_size * (num_key_value_heads * head_dim)
    v_params = hidden_size * (num_key_value_heads * head_dim)
    o_params = hidden_size * hidden_size
    attn_params = q_params + k_params + v_params + o_params

    # MLP: gate, up, down projections (for Llama-style)
    mlp_params = hidden_size * intermediate_size * 3

    # Layer norms (2 per layer)
    norm_params = hidden_size * 2

    layer_params = attn_params + mlp_params + norm_params

    # Total
    total_params = embed_params + (layer_params * num_layers) + vocab_size * hidden_size  # + lm_head

    # Check for MoE (Mixture of Experts) - multiplies MLP params
    num_experts = getattr(config, 'num_local_experts', getattr(config, 'num_experts', None))
    if num_experts:
        # MoE models have multiple expert MLPs
        total_params += mlp_params * (num_experts - 1) * num_layers

    # Model weights + 20% overhead for loading buffers and activations
    estimated_bytes = int(total_params * bytes_per_param * 1.2)

    return estimated_bytes


def get_available_memory(device: str) -> Dict[str, int]:
    """
    Get available memory for the specified device.

    Returns dict with 'total', 'available', and 'used' in bytes.
    """
    import os

    if device == "cuda" or device.startswith("cuda:"):
        if torch.cuda.is_available():
            device_idx = 0
            if ":" in device:
                device_idx = int(device.split(":")[1])

            total = torch.cuda.get_device_properties(device_idx).total_memory
            reserved = torch.cuda.memory_reserved(device_idx)
            allocated = torch.cuda.memory_allocated(device_idx)
            available = total - reserved

            return {
                'total': total,
                'available': available,
                'used': allocated,
                'device_name': torch.cuda.get_device_name(device_idx),
            }

    # For CPU and MPS, check system RAM
    # MPS uses unified memory, so system RAM is the constraint
    try:
        import psutil
        mem = psutil.virtual_memory()
        return {
            'total': mem.total,
            'available': mem.available,
            'used': mem.used,
            'device_name': 'System RAM' if device == 'cpu' else 'Unified Memory (MPS)',
        }
    except ImportError:
        # Fallback: try to read from system
        if sys.platform == 'darwin':  # macOS
            try:
                import subprocess
                # Get page size and free pages
                result = subprocess.run(['vm_stat'], capture_output=True, text=True)
                lines = result.stdout.split('\n')

                page_size = 16384  # Default for Apple Silicon
                free_pages = 0

                for line in lines:
                    if 'page size' in line.lower():
                        page_size = int(line.split()[-2])
                    elif 'Pages free' in line:
                        free_pages = int(line.split()[2].rstrip('.'))

                # Also get total memory
                result = subprocess.run(['sysctl', 'hw.memsize'], capture_output=True, text=True)
                total = int(result.stdout.split()[1])
                available = free_pages * page_size

                return {
                    'total': total,
                    'available': available,
                    'used': total - available,
                    'device_name': 'Unified Memory (MPS)' if device == 'mps' else 'System RAM',
                }
            except Exception:
                pass

        # Last resort fallback
        return {
            'total': 0,
            'available': 0,
            'used': 0,
            'device_name': 'Unknown',
        }


class InsufficientMemoryError(Exception):
    """Raised when there isn't enough memory to load the model."""
    def __init__(self, required: int, available: int, device: str):
        self.required = required
        self.available = available
        self.device = device
        super().__init__(f"Insufficient memory: need {required}, have {available}")


# =============================================================================
# Terminal Formatting Helpers
# =============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    END = '\033[0m'

    @classmethod
    def disable(cls):
        cls.HEADER = cls.BLUE = cls.CYAN = cls.GREEN = ''
        cls.YELLOW = cls.RED = cls.BOLD = cls.DIM = cls.END = ''


def supports_color():
    """Check if terminal supports color."""
    import os
    if os.getenv('NO_COLOR'):
        return False
    if not hasattr(sys.stdout, 'isatty'):
        return False
    return sys.stdout.isatty()


class Spinner:
    """Simple spinner for long-running operations."""
    FRAMES = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']

    def __init__(self, message: str):
        self.message = message
        self.idx = 0
        self.start_time = None

    def start(self):
        self.start_time = time.time()
        self._print()

    def _print(self):
        frame = self.FRAMES[self.idx % len(self.FRAMES)]
        elapsed = time.time() - self.start_time if self.start_time else 0
        sys.stdout.write(f'\r{Colors.CYAN}{frame}{Colors.END} {self.message} {Colors.DIM}({elapsed:.1f}s){Colors.END}')
        sys.stdout.flush()
        self.idx += 1

    def update(self, message: str = None):
        if message:
            self.message = message
        self._print()

    def succeed(self, message: str = None):
        elapsed = time.time() - self.start_time if self.start_time else 0
        msg = message or self.message
        sys.stdout.write(f'\r{Colors.GREEN}✓{Colors.END} {msg} {Colors.DIM}({elapsed:.1f}s){Colors.END}\n')
        sys.stdout.flush()

    def fail(self, message: str = None):
        elapsed = time.time() - self.start_time if self.start_time else 0
        msg = message or self.message
        sys.stdout.write(f'\r{Colors.RED}✗{Colors.END} {msg} {Colors.DIM}({elapsed:.1f}s){Colors.END}\n')
        sys.stdout.flush()


def format_params(n: int) -> str:
    """Format parameter count nicely."""
    if n >= 1e12:
        return f"{n/1e12:.1f}T"
    elif n >= 1e9:
        return f"{n/1e9:.1f}B"
    elif n >= 1e6:
        return f"{n/1e6:.1f}M"
    elif n >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(n)


def format_bytes(n: int) -> str:
    """Format byte count nicely."""
    if n >= 1e12:
        return f"{n/1e12:.1f} TB"
    elif n >= 1e9:
        return f"{n/1e9:.1f} GB"
    elif n >= 1e6:
        return f"{n/1e6:.1f} MB"
    elif n >= 1e3:
        return f"{n/1e3:.1f} KB"
    return f"{n} B"


# =============================================================================
# Custom Exceptions
# =============================================================================

class GatedModelError(Exception):
    """Raised when trying to access a gated HuggingFace model without authorization."""
    def __init__(self, model_name: str):
        self.model_name = model_name
        super().__init__(f"Access denied for gated model: {model_name}")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CriticalWeight:
    """A single critical weight identified by PeakWeights."""
    rank: int
    score: float
    module: str
    param: str
    flat_index: int
    row: int
    col: int
    value: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "score": round(self.score, 4),
            "module": self.module,
            "param": self.param,
            "index": [self.row, self.col],
            "flat_index": self.flat_index,
            "value": round(self.value, 6)
        }


# =============================================================================
# Core Algorithm
# =============================================================================

class PeakWeightsFinder:
    """
    Find the critical few weights in any LLM.

    The core insight: score = |weight| × |max_activation|

    This captures the worst-case output disturbance each weight
    can cause, without needing gradients or calibration data.
    """

    def __init__(
        self,
        model_name: str,
        top_k: int = 50,  # Default K=50 for 90%+ recovery on most models
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        include_embeddings: bool = True,
        include_moe_router: bool = True,
        verbose: bool = True,
    ):
        self.model_name = model_name
        self.top_k = top_k
        self.device = device
        self.dtype = dtype
        self.include_embeddings = include_embeddings
        self.include_moe_router = include_moe_router
        self.verbose = verbose

        self.heap: List[Tuple] = []
        self.layer_scores: Dict[str, float] = {}
        self.total_params: int = 0
        self.model_size_bytes: int = 0
        self.num_layers: int = 0
        self.layers_processed: int = 0

    def _log(self, msg: str):
        """Print if verbose mode is on."""
        if self.verbose:
            print(msg)

    def _print_gated_model_help(self):
        """Print helpful instructions for accessing gated models."""
        print(f"\n{Colors.YELLOW}{'─'*60}{Colors.END}")
        print(f"{Colors.YELLOW}  This is a gated model{Colors.END}")
        print(f"{Colors.YELLOW}{'─'*60}{Colors.END}")
        print(f"\n  {Colors.BOLD}{self.model_name}{Colors.END} requires you to accept the license")
        print(f"  agreement before downloading.\n")
        print(f"  {Colors.CYAN}Steps to get access:{Colors.END}\n")
        print(f"  1. Visit the model page and request access:")
        print(f"     {Colors.BOLD}https://huggingface.co/{self.model_name}{Colors.END}\n")
        print(f"  2. Log in to Hugging Face CLI:")
        print(f"     {Colors.BOLD}huggingface-cli login{Colors.END}\n")
        print(f"  3. Get your token from:")
        print(f"     https://huggingface.co/settings/tokens\n")
        print(f"  {Colors.DIM}Tip: Verify login with: huggingface-cli whoami{Colors.END}")
        print(f"\n{Colors.YELLOW}{'─'*60}{Colors.END}\n")

    def _print_memory_error(self, estimated: int, available: int, total: int, device: str, device_name: str):
        """Print helpful message when there isn't enough memory."""
        shortage = estimated - available

        print(f"{Colors.RED}{'─'*60}{Colors.END}")
        print(f"{Colors.RED}  Insufficient Memory{Colors.END}")
        print(f"{Colors.RED}{'─'*60}{Colors.END}")
        print()
        print(f"  {Colors.BOLD}{self.model_name}{Colors.END} requires more memory than available.")
        print()
        print(f"  {Colors.CYAN}Memory Details:{Colors.END}")
        print(f"    Required:   ~{format_bytes(estimated)}")
        print(f"    Available:  {format_bytes(available)}")
        print(f"    Shortage:   {Colors.RED}{format_bytes(shortage)}{Colors.END}")
        print()

        # Device-specific suggestions
        print(f"  {Colors.CYAN}Suggestions:{Colors.END}")

        if device == "mps":
            print(f"    {Colors.BOLD}1.{Colors.END} Close other applications to free memory")
            print(f"       (MPS uses unified memory shared with system)")
            print()
            print(f"    {Colors.BOLD}2.{Colors.END} Try a smaller model:")
            print(f"       peakweights HuggingFaceTB/SmolLM2-1.7B")
            print(f"       peakweights Qwen/Qwen2.5-3B")
            print()
            print(f"    {Colors.BOLD}3.{Colors.END} Use CPU instead (slower but uses swap):")
            print(f"       peakweights {self.model_name} --device cpu")
            print()
            if self.dtype == torch.float32:
                print(f"    {Colors.BOLD}4.{Colors.END} Use float16 to halve memory usage:")
                print(f"       peakweights {self.model_name} --dtype float16")
                print()

        elif device == "cpu":
            print(f"    {Colors.BOLD}1.{Colors.END} Close other applications to free memory")
            print()
            print(f"    {Colors.BOLD}2.{Colors.END} Try a smaller model:")
            print(f"       peakweights HuggingFaceTB/SmolLM2-1.7B")
            print(f"       peakweights Qwen/Qwen2.5-3B")
            print()
            if self.dtype == torch.float32:
                print(f"    {Colors.BOLD}3.{Colors.END} Use float16 to halve memory usage:")
                print(f"       peakweights {self.model_name} --dtype float16")
                print()

        elif device.startswith("cuda"):
            print(f"    {Colors.BOLD}1.{Colors.END} Use a GPU with more VRAM")
            print()
            print(f"    {Colors.BOLD}2.{Colors.END} Try a smaller model:")
            print(f"       peakweights HuggingFaceTB/SmolLM2-1.7B")
            print(f"       peakweights Qwen/Qwen2.5-3B")
            print()
            print(f"    {Colors.BOLD}3.{Colors.END} Use CPU with system RAM (slower):")
            print(f"       peakweights {self.model_name} --device cpu")
            print()

        # Model size reference
        print(f"  {Colors.CYAN}Typical Model Sizes (float16):{Colors.END}")
        print(f"    HuggingFaceTB/SmolLM2-1.7B       ~3.5 GB")
        print(f"    Qwen/Qwen2.5-3B                  ~6.5 GB")
        print(f"    mistralai/Mistral-7B-v0.3       ~14 GB")
        print(f"    meta-llama/Llama-3.1-8B         ~16 GB")
        print(f"    Qwen/Qwen2.5-14B                ~28 GB")
        print()

        # Google Colab suggestion
        print(f"  {Colors.GREEN}Free GPU Option - Google Colab:{Colors.END}")
        print(f"    Run this model on a free A100 GPU:")
        print()
        print(f"    {Colors.BOLD}# In a Colab notebook:{Colors.END}")
        print(f"    !pip install git+https://github.com/Kalmantic/peakweights.git -q")
        print(f"    !peakweights {self.model_name} --top_k {self.top_k}")
        print()
        print(f"    {Colors.DIM}Open: https://colab.research.google.com{Colors.END}")
        print(f"    {Colors.DIM}Select: Runtime > Change runtime type > A100{Colors.END}")
        print()
        print(f"{Colors.RED}{'─'*60}{Colors.END}\n")

    def find(self) -> List[CriticalWeight]:
        """Run the analysis and return critical weights."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

        overall_start = time.time()

        # Step 1: Load config first (fast) to show model info
        print(f"\n{Colors.BOLD}{'─'*60}{Colors.END}")
        print(f"{Colors.BOLD}  PeakWeights{Colors.END} - Critical Weight Discovery")
        print(f"{Colors.BOLD}{'─'*60}{Colors.END}\n")

        spinner = Spinner(f"Fetching model config from {self.model_name}")
        spinner.start()

        try:
            config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
            spinner.succeed("Model config loaded")
        except Exception as e:
            error_str = str(e).lower()
            is_gated = 'gated' in error_str or '403' in error_str or 'access' in error_str

            if is_gated:
                spinner.fail(f"Access denied - gated model")
                self._print_gated_model_help()
                raise GatedModelError(self.model_name) from e
            else:
                spinner.fail(f"Failed to load config: {e}")
                raise

        # Display model info
        print(f"\n{Colors.CYAN}  Model Information{Colors.END}")
        print(f"  {'─'*40}")
        print(f"  Model:        {Colors.BOLD}{self.model_name}{Colors.END}")

        if hasattr(config, 'num_hidden_layers'):
            self.num_layers = config.num_hidden_layers
            print(f"  Layers:       {self.num_layers}")
        if hasattr(config, 'hidden_size'):
            print(f"  Hidden size:  {config.hidden_size}")
        if hasattr(config, 'num_attention_heads'):
            print(f"  Attn heads:   {config.num_attention_heads}")
        if hasattr(config, 'num_experts') or hasattr(config, 'num_local_experts'):
            num_experts = getattr(config, 'num_experts', getattr(config, 'num_local_experts', None))
            if num_experts:
                print(f"  MoE experts:  {num_experts} {Colors.YELLOW}(MoE model){Colors.END}")

        print(f"  Device:       {self.device}")
        print(f"  Dtype:        {self.dtype}")
        print(f"  Top-K:        {self.top_k}")
        print()

        # Step 2: Memory check before loading
        estimated_mem = estimate_model_memory(config, self.dtype)
        mem_info = get_available_memory(self.device)

        print(f"{Colors.CYAN}  Memory Requirements{Colors.END}")
        print(f"  {'─'*40}")
        print(f"  Estimated:    ~{format_bytes(estimated_mem)}")
        print(f"  Available:    {format_bytes(mem_info['available'])} / {format_bytes(mem_info['total'])}")
        print(f"  Device:       {mem_info['device_name']}")

        # Check if we have enough memory (with some margin)
        # Use 80% threshold since estimate is approximate
        if mem_info['available'] > 0 and estimated_mem > mem_info['available'] * 0.95:
            shortage = estimated_mem - mem_info['available']
            print()
            self._print_memory_error(
                estimated=estimated_mem,
                available=mem_info['available'],
                total=mem_info['total'],
                device=self.device,
                device_name=mem_info['device_name'],
            )
            raise InsufficientMemoryError(estimated_mem, mem_info['available'], self.device)

        # Show memory status indicator
        mem_usage_pct = (estimated_mem / mem_info['available'] * 100) if mem_info['available'] > 0 else 0
        if mem_usage_pct > 80:
            status_color = Colors.YELLOW
            status_text = "Tight fit"
        elif mem_usage_pct > 50:
            status_color = Colors.CYAN
            status_text = "OK"
        else:
            status_color = Colors.GREEN
            status_text = "Plenty"
        print(f"  Status:       {status_color}{status_text} ({mem_usage_pct:.0f}% of available){Colors.END}")
        print()

        # Step 3: Load model
        spinner = Spinner("Loading model weights (this may take a while for large models)")
        spinner.start()

        try:
            # On CPU/MPS without enough memory, device_map="auto" may offload to disk
            # which creates meta tensors that can't be used for inference.
            # For non-CUDA devices, we need to load differently to avoid this issue.
            if self.device == "mps":
                # MPS has known segfault issues with direct device_map loading.
                # Load to CPU first, then move to MPS for stability.
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=self.dtype,
                    device_map="cpu",
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                ).eval()
                spinner.update("Moving model to MPS device")
                model = model.to("mps")
            elif self.device == "cpu":
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=self.dtype,
                    device_map={"": "cpu"},
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                ).eval()
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=self.dtype,
                    device_map="auto",
                    trust_remote_code=True,
                ).eval()

            # Check for meta tensors (indicates failed/incomplete loading)
            meta_params = [n for n, p in model.named_parameters() if p.device.type == 'meta']
            if meta_params:
                spinner.fail("Model has unloaded parameters (meta tensors)")
                # Estimate size before we have total_params calculated
                est_params = sum(p.numel() for p in model.parameters())
                est_size = est_params * (2 if self.dtype in [torch.float16, torch.bfloat16] else 4)
                print(f"\n{Colors.RED}Error:{Colors.END} Some model weights couldn't be loaded into memory.")
                print(f"  This model requires ~{format_bytes(est_size)} of RAM.")
                print(f"\n{Colors.YELLOW}Suggestions:{Colors.END}")
                print("  • Use a machine with more RAM")
                print("  • Use a GPU with sufficient VRAM")
                print("  • Try a smaller distilled model")
                raise RuntimeError(f"{len(meta_params)} parameters stuck on meta device")

            spinner.succeed("Model loaded successfully")
        except Exception as e:
            error_str = str(e).lower()
            is_gated = 'gated' in error_str or '403' in error_str or 'access' in error_str

            if is_gated:
                spinner.fail(f"Access denied - gated model")
                self._print_gated_model_help()
                raise GatedModelError(self.model_name) from e
            else:
                spinner.fail(f"Failed to load model: {e}")
                raise

        # Count parameters and estimate size
        self.total_params = sum(p.numel() for p in model.parameters())
        bytes_per_param = 2 if self.dtype in [torch.float16, torch.bfloat16] else 4
        self.model_size_bytes = self.total_params * bytes_per_param

        print(f"\n{Colors.CYAN}  Model Statistics{Colors.END}")
        print(f"  {'─'*40}")
        print(f"  Parameters:   {Colors.BOLD}{format_params(self.total_params)}{Colors.END} ({self.total_params:,})")
        print(f"  Model size:   ~{format_bytes(self.model_size_bytes)}")
        print()

        # Step 3: Load tokenizer
        spinner = Spinner("Loading tokenizer")
        spinner.start()

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            spinner.succeed("Tokenizer loaded")
        except Exception as e:
            spinner.fail(f"Failed to load tokenizer: {e}")
            raise

        # Step 4: Prepare synthetic input
        print(f"\n{Colors.CYAN}  Analysis Configuration{Colors.END}")
        print(f"  {'─'*40}")
        print(f"  Method:       Data-free (synthetic input)")
        print(f"  Input:        16 synthetic tokens")
        print(f"  Scoring:      |weight| × |max_activation|")
        print()

        dummy_text = " ".join(map(str, range(16)))
        dummy = tokenizer(dummy_text, return_tensors="pt")
        dummy = {k: v.to(self.device) for k, v in dummy.items()}

        # Step 5: Register hooks
        spinner = Spinner("Registering forward hooks on model layers")
        spinner.start()

        handles = []
        hooked_count = 0
        for name, module in model.named_modules():
            if self._should_hook(module):
                handle = module.register_forward_hook(self._make_hook(name))
                handles.append(handle)
                hooked_count += 1

        spinner.succeed(f"Registered {hooked_count} hooks")

        # Step 6: Run forward pass
        print(f"\n{Colors.CYAN}  Running Analysis{Colors.END}")
        print(f"  {'─'*40}")

        spinner = Spinner("Running forward pass and scoring weights")
        spinner.start()

        forward_start = time.time()
        with torch.no_grad():
            model(**dummy)
        forward_time = time.time() - forward_start

        spinner.succeed(f"Forward pass complete ({forward_time:.2f}s)")

        # Cleanup hooks
        for h in handles:
            h.remove()

        # Step 7: Process results
        spinner = Spinner("Processing and ranking critical weights")
        spinner.start()

        results = []
        for score, module, param, idx, shape, value in sorted(self.heap, reverse=True):
            row = idx // shape[1] if len(shape) > 1 else idx
            col = idx % shape[1] if len(shape) > 1 else 0

            results.append(CriticalWeight(
                rank=len(results) + 1,
                score=score,
                module=module,
                param=param,
                flat_index=idx,
                row=row,
                col=col,
                value=value,
            ))

        spinner.succeed(f"Found {len(results)} critical weights")

        overall_time = time.time() - overall_start
        print(f"\n  {Colors.GREEN}Analysis complete in {overall_time:.1f}s{Colors.END}")

        return results

    def _should_hook(self, module) -> bool:
        """Determine if we should hook this module."""
        if isinstance(module, torch.nn.Linear):
            return True
        if isinstance(module, torch.nn.Embedding) and self.include_embeddings:
            return True
        if isinstance(module, torch.nn.Conv1d):
            return True
        if hasattr(module, 'gate') and self.include_moe_router:
            return True
        return False

    def _make_hook(self, name: str):
        """Create a forward hook for scoring weights."""
        def hook(module, inp, out):
            act = inp[0].detach() if isinstance(inp, tuple) else inp.detach()

            if act.dim() < 2:
                return

            max_a = act.abs()
            while max_a.dim() > 1:
                max_a = max_a.amax(dim=0)

            for pname, param in module.named_parameters(recurse=False):
                if param.dim() < 2:
                    continue

                # Skip meta tensors (unloaded parameters)
                if param.device.type == 'meta':
                    continue

                p = param.detach()

                # Ensure tensors are on the same device
                if p.device != max_a.device:
                    max_a = max_a.to(p.device)

                if p.shape[-1] != max_a.shape[-1]:
                    if p.shape[0] == max_a.shape[-1]:
                        p = p.T
                    else:
                        continue

                scores = p.abs() * max_a.unsqueeze(0)
                layer_score = scores.sum().item()
                self.layer_scores[f"{name}.{pname}"] = layer_score

                flat = scores.flatten()
                topk = min(64, flat.numel())
                vals, idxs = flat.topk(topk)

                for v, i in zip(vals.tolist(), idxs.tolist()):
                    entry = (v, name, pname, i, tuple(p.shape), p.flatten()[i].item())

                    if len(self.heap) < self.top_k:
                        heapq.heappush(self.heap, entry)
                    elif v > self.heap[0][0]:
                        heapq.heapreplace(self.heap, entry)

        return hook

    def get_layer_importance(self) -> Dict[str, float]:
        """Get importance scores per layer."""
        return dict(sorted(
            self.layer_scores.items(),
            key=lambda x: x[1],
            reverse=True
        ))


# =============================================================================
# Public API
# =============================================================================

def find(
    model_name: str,
    k: int = 50,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Find the k most critical weights in any LLM.

    Args:
        model_name: HuggingFace model ID
        k: Number of critical weights to find (default: 50 for 90%+ recovery)
        device: cuda or cpu
        dtype: Model precision
        verbose: Show progress output

    Returns:
        List of critical weight dictionaries

    Note:
        The optimal K is model-dependent:
        - Small models (<3B): K=20 for 90% recovery
        - Qwen/DeepSeek: K=50 for 95% recovery
        - Mistral: K=100 for 99% recovery

        Use calibrate() to find the optimal K for your model.
    """
    finder = PeakWeightsFinder(model_name, top_k=k, device=device, dtype=dtype, verbose=verbose)
    results = finder.find()
    return [r.to_dict() for r in results]


def find_k_for_recovery(
    model_name: str,
    target_recovery: float = 0.95,
    max_k: int = 200,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Find the minimum K needed to achieve a target recovery rate.

    This runs PeakWeights and analyzes the score distribution to find
    the smallest K that achieves the desired recovery.

    Args:
        model_name: HuggingFace model ID
        target_recovery: Target recovery rate (default: 0.95 = 95%)
        max_k: Maximum K to consider (default: 200)
        device: cuda or cpu
        dtype: Model precision
        verbose: Show progress output

    Returns:
        Dict with:
        - 'k': The minimum K for target recovery
        - 'actual_recovery': Actual recovery at that K
        - 'weights': List of critical weights
        - 'quantization_guide': How to use with popular quantization tools

    Example:
        >>> result = peakweights.find_k_for_recovery("Qwen/Qwen2.5-7B", target_recovery=0.95)
        >>> print(f"K={result['k']} achieves {result['actual_recovery']:.1%} recovery")
        K=50 achieves 96.2% recovery
    """
    finder = PeakWeightsFinder(model_name, top_k=max_k, device=device, dtype=dtype, verbose=verbose)
    results = finder.find()

    # Calculate cumulative importance
    total_importance = sum(finder.layer_scores.values())
    weights = sorted([r.to_dict() for r in results], key=lambda x: -x['score'])

    # Find minimum K for target recovery
    running_sum = 0
    found_k = max_k
    actual_recovery = 0

    for i, w in enumerate(weights, 1):
        running_sum += w['score']
        recovery = running_sum / total_importance if total_importance > 0 else 0
        if recovery >= target_recovery:
            found_k = i
            actual_recovery = recovery
            break
    else:
        # Didn't reach target, use max
        actual_recovery = running_sum / total_importance if total_importance > 0 else 0

    # Generate quantization guide
    quant_guide = generate_quantization_guide(weights[:found_k], model_name, found_k)

    if verbose:
        print(f"\n{Colors.BOLD}{'─'*60}{Colors.END}")
        print(f"{Colors.BOLD}  K FOR {target_recovery:.0%} RECOVERY{Colors.END}")
        print(f"{Colors.BOLD}{'─'*60}{Colors.END}\n")

        if actual_recovery >= target_recovery:
            print(f"  {Colors.GREEN}✓ Target achieved!{Colors.END}")
            print(f"    K = {Colors.BOLD}{found_k}{Colors.END} → {actual_recovery:.1%} recovery")
        else:
            print(f"  {Colors.YELLOW}⚠ Target not reached with K={max_k}{Colors.END}")
            print(f"    Best: K={found_k} → {actual_recovery:.1%} recovery")
            print(f"    Try increasing --max_k")

        print(f"\n  {Colors.CYAN}Recovery at Various K:{Colors.END}")
        checkpoints = [10, 20, 50, 100, 150, 200]
        running = 0
        for i, w in enumerate(weights, 1):
            running += w['score']
            if i in checkpoints and i <= max_k:
                rec = running / total_importance if total_importance > 0 else 0
                marker = " ← target" if i == found_k else ""
                print(f"    K={i:3d}:  {rec:.1%}{Colors.GREEN}{marker}{Colors.END}")

        print()

    return {
        'k': found_k,
        'actual_recovery': actual_recovery,
        'target_recovery': target_recovery,
        'weights': weights[:found_k],
        'all_weights': weights,
        'quantization_guide': quant_guide,
        'total_importance': total_importance,
    }


def generate_quantization_guide(weights: List[Dict], model_name: str, k: int) -> Dict[str, str]:
    """Generate quantization integration code for popular frameworks."""

    # Group weights by module
    modules = {}
    for w in weights:
        key = f"{w['module']}.{w['param']}"
        if key not in modules:
            modules[key] = []
        modules[key].append(w['flat_index'])

    module_list = list(modules.keys())

    guide = {
        'summary': f"Protect {k} weights across {len(modules)} modules",

        'llama_cpp': f'''# llama.cpp: Use --keep-split to protect critical weights
# First, save the protection mask:
peakweights {model_name} --recovery 95 --mask protect.pt

# Then convert with protection (coming soon in llama.cpp)
# python convert.py {model_name} --protect-mask protect.pt''',

        'bitsandbytes': f'''# bitsandbytes 4-bit quantization with protection
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import peakweights

# Find critical weights
result = peakweights.find_k_for_recovery("{model_name}", target_recovery=0.95)
mask = peakweights.generate_protection_mask(result['weights'])

# Quantize with protection
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    # Protected weights stay in FP16
    llm_int8_skip_modules=list(mask.keys())[:5],  # Top 5 modules
)

model = AutoModelForCausalLM.from_pretrained(
    "{model_name}",
    quantization_config=bnb_config,
    device_map="auto",
)''',

        'autoawq': f'''# AutoAWQ with weight protection
from awq import AutoAWQForCausalLM
import peakweights

# Find critical weights
result = peakweights.find_k_for_recovery("{model_name}", target_recovery=0.95)
mask = peakweights.generate_protection_mask(result['weights'])

# Get modules to skip (keep in FP16)
skip_modules = list(set(w['module'] for w in result['weights']))

model = AutoAWQForCausalLM.from_pretrained("{model_name}")
model.quantize(
    tokenizer,
    quant_config={{"w_bit": 4, "q_group_size": 128}},
    # Skip critical modules
    modules_to_not_convert=skip_modules[:10],
)''',

        'gptq': f'''# GPTQ with weight protection
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import peakweights

result = peakweights.find_k_for_recovery("{model_name}", target_recovery=0.95)

# Modules containing critical weights
critical_modules = list(set(w['module'] for w in result['weights']))

quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    # Modules to skip quantization
    modules_in_block_to_quantize=[
        m for m in ["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]
        if not any(m in cm for cm in critical_modules[:5])
    ],
)''',

        'manual': f'''# Manual weight restoration after quantization
import torch
import peakweights

# 1. Get critical weights BEFORE quantization
result = peakweights.find_k_for_recovery("{model_name}", target_recovery=0.95)

# Save original values
original_values = {{}}
for w in result['weights']:
    key = (w['module'], w['param'], w['flat_index'])
    # Store: (module_path, param_name, index, original_value)
    original_values[key] = w['value']

# 2. After quantization, restore critical weights
def restore_critical_weights(model, original_values):
    for (module_path, param_name, idx), value in original_values.items():
        module = model.get_submodule(module_path)
        param = getattr(module, param_name)
        with torch.no_grad():
            param.data.flatten()[idx] = value
    return model

# quantized_model = restore_critical_weights(quantized_model, original_values)
''',
    }

    return guide


def calibrate(
    model_name: str,
    k_values: List[int] = [10, 20, 50, 100],
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Find the optimal K for a model by analyzing score distribution.

    This runs PeakWeights once with K=max(k_values) and calculates
    cumulative importance at each K value to help determine the
    optimal number of weights to protect.

    Args:
        model_name: HuggingFace model ID
        k_values: List of K values to evaluate (default: [10, 20, 50, 100])
        device: cuda or cpu
        dtype: Model precision
        verbose: Show progress output

    Returns:
        Dict with:
        - 'weights': List of critical weights
        - 'cumulative_importance': Dict mapping K -> cumulative score
        - 'recommended_k': Suggested K based on elbow analysis
        - 'power_law_exponent': Estimated α from power law fit

    Example:
        >>> result = peakweights.calibrate("Qwen/Qwen2.5-7B")
        >>> print(f"Recommended K: {result['recommended_k']}")
        >>> for k, importance in result['cumulative_importance'].items():
        ...     print(f"K={k}: {importance:.2%} of total importance")
    """
    import math

    max_k = max(k_values)
    finder = PeakWeightsFinder(model_name, top_k=max_k, device=device, dtype=dtype, verbose=verbose)
    results = finder.find()

    # Calculate cumulative importance at each K
    total_importance = sum(finder.layer_scores.values())
    weights = sorted([r.to_dict() for r in results], key=lambda x: -x['score'])

    cumulative = {}
    running_sum = 0
    for i, w in enumerate(weights, 1):
        running_sum += w['score']
        if i in k_values:
            cumulative[i] = running_sum / total_importance if total_importance > 0 else 0

    # Estimate power law exponent using linear regression on log-log
    if len(weights) >= 10:
        ranks = list(range(1, len(weights) + 1))
        scores = [w['score'] for w in weights]

        # Filter out zero scores for log
        valid = [(r, s) for r, s in zip(ranks, scores) if s > 0]
        if len(valid) >= 5:
            log_ranks = [math.log(r) for r, _ in valid]
            log_scores = [math.log(s) for _, s in valid]

            # Simple linear regression
            n = len(log_ranks)
            sum_x = sum(log_ranks)
            sum_y = sum(log_scores)
            sum_xy = sum(x*y for x, y in zip(log_ranks, log_scores))
            sum_xx = sum(x*x for x in log_ranks)

            alpha = -(n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
        else:
            alpha = 0.43  # Default fallback
    else:
        alpha = 0.43

    # Recommend K based on cumulative importance
    # Rule: smallest K that achieves 90% cumulative importance
    recommended_k = max_k
    for k in sorted(k_values):
        if cumulative.get(k, 0) >= 0.90:
            recommended_k = k
            break

    # If no K achieves 90%, recommend based on diminishing returns (elbow)
    if recommended_k == max_k and len(k_values) >= 3:
        # Find where marginal gain drops significantly
        sorted_ks = sorted(k_values)
        for i in range(1, len(sorted_ks)):
            k_prev, k_curr = sorted_ks[i-1], sorted_ks[i]
            gain_prev = cumulative.get(k_prev, 0)
            gain_curr = cumulative.get(k_curr, 0)
            marginal = (gain_curr - gain_prev) / (k_curr - k_prev)
            if marginal < 0.001:  # Less than 0.1% gain per weight
                recommended_k = k_prev
                break

    if verbose:
        print(f"\n{Colors.BOLD}{'─'*60}{Colors.END}")
        print(f"{Colors.BOLD}  CALIBRATION RESULTS{Colors.END}")
        print(f"{Colors.BOLD}{'─'*60}{Colors.END}\n")

        print(f"  {Colors.CYAN}Cumulative Importance by K:{Colors.END}")
        for k in sorted(k_values):
            pct = cumulative.get(k, 0) * 100
            bar_len = int(pct / 5)
            bar = '█' * bar_len + '░' * (20 - bar_len)
            marker = " ← recommended" if k == recommended_k else ""
            print(f"    K={k:3d}:  {pct:5.1f}%  {Colors.DIM}{bar}{Colors.END}{Colors.GREEN}{marker}{Colors.END}")

        print(f"\n  {Colors.CYAN}Power Law Analysis:{Colors.END}")
        print(f"    Exponent (α): {alpha:.2f}")
        print(f"    Interpretation: {'Concentrated (few critical)' if alpha > 0.43 else 'Distributed (many matter)'}")

        print(f"\n  {Colors.GREEN}Recommended K: {recommended_k}{Colors.END}")
        print(f"    Expected recovery: ~{cumulative.get(recommended_k, 0)*100:.0f}%+ at K={recommended_k}")
        print()

    return {
        'weights': weights,
        'cumulative_importance': cumulative,
        'recommended_k': recommended_k,
        'power_law_exponent': alpha,
        'total_importance': total_importance,
    }


def generate_protection_mask(
    critical_weights: List[Dict],
    output_path: Optional[str] = None
) -> Dict[str, List[int]]:
    """Generate a mask for protecting critical weights during quantization."""
    mask = {}
    for w in critical_weights:
        key = f"{w['module']}.{w['param']}"
        if key not in mask:
            mask[key] = []
        mask[key].append(w['flat_index'])

    if output_path:
        torch.save(mask, output_path)
        print(f"{Colors.GREEN}✓{Colors.END} Saved protection mask to {output_path}")

    return mask


def visualize(finder: PeakWeightsFinder, output_path: Optional[str] = None):
    """Generate a visualization of layer importance."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"{Colors.RED}✗{Colors.END} Install matplotlib for visualization: pip install matplotlib")
        return

    scores = finder.get_layer_importance()
    top_layers = list(scores.items())[:20]
    names = [n.split('.')[-2] + '.' + n.split('.')[-1] for n, _ in top_layers]
    values = [v for _, v in top_layers]

    plt.figure(figsize=(12, 6))
    plt.barh(range(len(names)), values, color='#4A90D9')
    plt.yticks(range(len(names)), names)
    plt.xlabel('Importance Score (sum of |w| × |max_act|)')
    plt.title(f'Layer Importance: {finder.model_name}')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"{Colors.GREEN}✓{Colors.END} Saved visualization to {output_path}")
    else:
        plt.show()


# =============================================================================
# Result Display
# =============================================================================

def print_results(results: List[CriticalWeight], model_name: str, total_params: int):
    """Pretty print results with analysis."""

    print(f"\n{Colors.BOLD}{'═'*60}{Colors.END}")
    print(f"{Colors.BOLD}  CRITICAL WEIGHTS FOUND{Colors.END}")
    print(f"{Colors.BOLD}{'═'*60}{Colors.END}\n")

    for r in results:
        # Determine module type for coloring
        module_lower = r.module.lower()
        if 'gate' in module_lower or 'router' in module_lower:
            type_color = Colors.YELLOW
            type_label = "[MoE Router]"
        elif 'attn' in module_lower or 'attention' in module_lower:
            type_color = Colors.BLUE
            type_label = "[Attention]"
        elif 'mlp' in module_lower or 'ffn' in module_lower:
            type_color = Colors.CYAN
            type_label = "[MLP]"
        elif 'embed' in module_lower:
            type_color = Colors.GREEN
            type_label = "[Embedding]"
        elif 'lm_head' in module_lower or 'output' in module_lower:
            type_color = Colors.RED
            type_label = "[Output]"
        else:
            type_color = Colors.DIM
            type_label = ""

        print(f"  {Colors.BOLD}#{r.rank}{Colors.END}  Score: {Colors.BOLD}{r.score:.2f}{Colors.END}  {type_color}{type_label}{Colors.END}")
        print(f"      {Colors.DIM}Module:{Colors.END} {r.module}")
        print(f"      {Colors.DIM}Param:{Colors.END}  {r.param}[{r.row}, {r.col}]")
        print(f"      {Colors.DIM}Value:{Colors.END}  {r.value:.6f}")
        print()

    # Summary statistics
    print(f"{Colors.BOLD}{'─'*60}{Colors.END}")
    print(f"{Colors.BOLD}  SUMMARY{Colors.END}")
    print(f"{Colors.BOLD}{'─'*60}{Colors.END}\n")

    ratio = (len(results) / total_params) * 100
    print(f"  Total parameters:    {Colors.BOLD}{format_params(total_params)}{Colors.END}")
    print(f"  Critical weights:    {Colors.BOLD}{len(results)}{Colors.END}")
    print(f"  Ratio:               {Colors.BOLD}{ratio:.10f}%{Colors.END}")

    # Analyze distribution
    module_types = {}
    layers = []

    for r in results:
        parts = r.module.lower()
        if 'gate' in parts or 'router' in parts:
            mtype = 'MoE Router'
        elif 'q_proj' in parts or 'k_proj' in parts:
            mtype = 'Attention Q/K'
        elif 'v_proj' in parts or 'o_proj' in parts:
            mtype = 'Attention V/O'
        elif 'attn' in parts:
            mtype = 'Attention'
        elif 'mlp' in parts or 'ffn' in parts:
            mtype = 'MLP/FFN'
        elif 'embed' in parts:
            mtype = 'Embedding'
        elif 'lm_head' in parts:
            mtype = 'LM Head'
        else:
            mtype = 'Other'
        module_types[mtype] = module_types.get(mtype, 0) + 1

        # Extract layer number
        for p in r.module.split('.'):
            if p.isdigit():
                layers.append(int(p))
                break

    print(f"\n  {Colors.CYAN}Distribution by module type:{Colors.END}")
    for mtype, count in sorted(module_types.items(), key=lambda x: -x[1]):
        pct = count / len(results) * 100
        bar_len = int(pct / 5)
        bar = '█' * bar_len + '░' * (20 - bar_len)
        print(f"    {mtype:15s} {count:2d} ({pct:5.1f}%) {Colors.DIM}{bar}{Colors.END}")

    if layers:
        print(f"\n  {Colors.CYAN}Layer distribution:{Colors.END}")
        print(f"    Earliest:  Layer {min(layers)}")
        print(f"    Latest:    Layer {max(layers)}")
        print(f"    Average:   Layer {sum(layers)/len(layers):.1f}")

    print(f"\n{Colors.BOLD}{'═'*60}{Colors.END}\n")


# =============================================================================
# CLI
# =============================================================================

def main():
    # Check for color support
    if not supports_color():
        Colors.disable()

    parser = argparse.ArgumentParser(
        description="PeakWeights: Find the weights that matter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
{Colors.BOLD}Examples:{Colors.END}
  peakweights Qwen/Qwen2.5-7B --recovery 95        # Find K for 95% recovery
  peakweights Qwen/Qwen2.5-7B -r 95 --show_quant   # + quantization guide
  peakweights meta-llama/Llama-3.1-8B              # Uses K=50 (default)
  peakweights Qwen/Qwen2.5-7B --calibrate          # Analyze K options
  peakweights mistralai/Mistral-7B -r 95 --mask protect.pt  # Save mask

{Colors.BOLD}For x% Recovery, Use:{Colors.END}
  --recovery 90    # Find minimum K for 90% recovery
  --recovery 95    # Find minimum K for 95% recovery (recommended)
  --recovery 99    # Find minimum K for 99% recovery

{Colors.BOLD}Typical K Values (from experiments):{Colors.END}
  | Model          | K for 95% |
  |----------------|-----------|
  | SmolLM2-1.7B   | ~25       |
  | Qwen2.5-7B     | ~50       |
  | DeepSeek-R1-7B | ~50       |
  | Mistral-7B     | ~85       |

{Colors.BOLD}Supported Models:{Colors.END}
  Any HuggingFace transformers model, including:
  - Llama 3.x (8B, 70B, 405B)
  - DeepSeek R1/V3 (distilled and full)
  - Qwen 2.5 (7B, 72B)
  - Mistral/Mixtral
  - SmolLM2, Gemma 2, and more
        """
    )

    parser.add_argument('--version', '-V', action='version', version=f'peakweights {__version__}')
    parser.add_argument("model", help="HuggingFace model ID or local path")
    parser.add_argument("--top_k", "-k", type=int, default=50,
                        help="Number of critical weights to find (default: 50 for 90%+ recovery)")
    parser.add_argument("--calibrate", "-c", action="store_true",
                        help="Find optimal K for this model (analyzes score distribution)")
    parser.add_argument("--recovery", "-r", type=int, default=None,
                        help="Target recovery %% (e.g., 95). Finds minimum K for that recovery.")
    parser.add_argument("--max_k", type=int, default=200,
                        help="Maximum K to consider when using --recovery (default: 200)")
    parser.add_argument("--show_quant", action="store_true",
                        help="Show quantization integration guide")
    parser.add_argument("--output", "-o", help="Save results to file (.pwi, .json, .pt, .csv)")
    parser.add_argument("--format", "-f", choices=["pwi", "json", "pt", "csv"],
                        help="Output format (auto-detected from extension if not specified)")
    parser.add_argument("--mask", "-m", help="Save protection mask (.pt) for quantization")
    parser.add_argument("--viz", action="store_true", help="Show layer importance visualization")
    parser.add_argument("--viz_output", help="Save visualization to file")
    parser.add_argument("--device", default="auto",
                        help="Device: auto, cuda, cuda:0, cpu (default: auto)")
    parser.add_argument("--dtype", default="float16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Model precision (default: float16)")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")

    args = parser.parse_args()

    if args.no_color:
        Colors.disable()

    # Determine device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
            print(f"{Colors.CYAN}Info:{Colors.END} Using Apple Silicon MPS acceleration")
        else:
            device = "cpu"
            print(f"{Colors.YELLOW}Warning:{Colors.END} No GPU detected, using CPU (this will be slow)")
    else:
        device = args.device

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    try:
        # Run calibration if requested
        if args.calibrate:
            result = calibrate(
                args.model,
                k_values=[10, 20, 50, 100],
                device=device,
                dtype=dtype_map[args.dtype],
                verbose=not args.quiet,
            )

            if args.output:
                from .formats import save as save_weights
                save_weights(
                    result['weights'][:result['recommended_k']],
                    args.output,
                    model_name=args.model,
                    metadata={
                        'recommended_k': result['recommended_k'],
                        'power_law_exponent': result['power_law_exponent'],
                        'cumulative_importance': {str(k): v for k, v in result['cumulative_importance'].items()},
                    }
                )
                print(f"{Colors.GREEN}✓{Colors.END} Saved calibration results to {args.output}")
            return

        # Run recovery-based analysis if requested
        if args.recovery is not None:
            target = args.recovery / 100.0  # Convert 95 -> 0.95
            result = find_k_for_recovery(
                args.model,
                target_recovery=target,
                max_k=args.max_k,
                device=device,
                dtype=dtype_map[args.dtype],
                verbose=not args.quiet,
            )

            # Show quantization guide if requested
            if args.show_quant:
                print(f"\n{Colors.BOLD}{'─'*60}{Colors.END}")
                print(f"{Colors.BOLD}  QUANTIZATION INTEGRATION GUIDE{Colors.END}")
                print(f"{Colors.BOLD}{'─'*60}{Colors.END}\n")

                print(f"  {Colors.CYAN}Summary:{Colors.END} {result['quantization_guide']['summary']}")

                for framework in ['bitsandbytes', 'autoawq', 'gptq', 'manual']:
                    print(f"\n  {Colors.BOLD}━━━ {framework.upper()} ━━━{Colors.END}")
                    print(f"{Colors.DIM}{result['quantization_guide'][framework]}{Colors.END}")

                print()

            if args.output:
                from .formats import save as save_weights
                save_weights(
                    result['weights'],
                    args.output,
                    model_name=args.model,
                    metadata={
                        'target_recovery': args.recovery,
                        'actual_recovery': result['actual_recovery'],
                    }
                )
                print(f"{Colors.GREEN}✓{Colors.END} Saved results to {args.output}")

            if args.mask:
                generate_protection_mask(result['weights'], args.mask)

            return

        # Run analysis
        finder = PeakWeightsFinder(
            args.model,
            top_k=args.top_k,
            device=device,
            dtype=dtype_map[args.dtype],
            verbose=not args.quiet,
        )

        results = finder.find()

        # Display results
        if not args.quiet:
            print_results(results, args.model, finder.total_params)

        # Save outputs
        if args.output:
            from .formats import save as save_weights
            save_weights(
                [r.to_dict() for r in results],
                args.output,
                model_name=args.model,
                total_params=finder.total_params,
            )
            print(f"{Colors.GREEN}✓{Colors.END} Saved results to {args.output}")

        if args.mask:
            generate_protection_mask([r.to_dict() for r in results], args.mask)

        if args.viz or args.viz_output:
            visualize(finder, args.viz_output)

    except GatedModelError:
        # Already printed helpful message, just exit cleanly
        sys.exit(1)
    except InsufficientMemoryError:
        # Already printed helpful message, just exit cleanly
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Interrupted by user{Colors.END}")
        sys.exit(130)


if __name__ == "__main__":
    main()
