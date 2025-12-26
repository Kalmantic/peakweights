"""
PeakWeights: Find the weights that matter. Protect them. Quantize the rest.

One-pass, data-free discovery of critical LLM parameters.

Usage:
    peakweights meta-llama/Llama-3.1-70B
    peakweights deepseek-ai/DeepSeek-V3 --top_k 10
    peakweights Qwen/Qwen2.5-72B --viz

License: MIT
"""

import torch
import heapq
import json
import argparse
import sys
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


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
        top_k: int = 10,
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

        # Step 2: Load model
        spinner = Spinner("Loading model weights (this may take a while for large models)")
        spinner.start()

        try:
            # On CPU/MPS without enough memory, device_map="auto" may offload to disk
            # which creates meta tensors that can't be used for inference.
            # For non-CUDA devices, we need to load differently to avoid this issue.
            if self.device in ("cpu", "mps"):
                # Load with explicit device placement - may fail on large models
                # but gives a clear error rather than meta tensor issues
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=self.dtype,
                    device_map={"": self.device},
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
    k: int = 10,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Find the k most critical weights in any LLM.

    Args:
        model_name: HuggingFace model ID
        k: Number of critical weights to find (default: 10)
        device: cuda or cpu
        dtype: Model precision
        verbose: Show progress output

    Returns:
        List of critical weight dictionaries
    """
    finder = PeakWeightsFinder(model_name, top_k=k, device=device, dtype=dtype, verbose=verbose)
    results = finder.find()
    return [r.to_dict() for r in results]


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
  peakweights meta-llama/Llama-3.1-8B
  peakweights deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --top_k 10
  peakweights Qwen/Qwen2.5-72B --output weights.json
  peakweights mistralai/Mistral-Large --mask protect.pt --viz

{Colors.BOLD}Supported Models:{Colors.END}
  Any HuggingFace transformers model, including:
  - Llama 3.x (8B, 70B, 405B)
  - DeepSeek R1/V3 (distilled and full)
  - Qwen 2.5 (7B, 72B)
  - Mistral/Mixtral
  - Gemma 2
        """
    )

    parser.add_argument("model", help="HuggingFace model ID or local path")
    parser.add_argument("--top_k", "-k", type=int, default=10,
                        help="Number of critical weights to find (default: 10)")
    parser.add_argument("--output", "-o", help="Save results to JSON file")
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
        with open(args.output, 'w') as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
        print(f"{Colors.GREEN}✓{Colors.END} Saved results to {args.output}")

    if args.mask:
        generate_protection_mask([r.to_dict() for r in results], args.mask)

    if args.viz or args.viz_output:
        visualize(finder, args.viz_output)


if __name__ == "__main__":
    main()
