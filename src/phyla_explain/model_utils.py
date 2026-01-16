"""Model utilities including critical residual connection fix"""
import sys, os

def apply_residual_fix():
    """
    CRITICAL BUG FIX: BiMambaWrapper.forward() missing residual connections.
    Without fix: CLS embeddings collapse to identical values
    With fix: r ≈ 0.54 correlation with phylogenetic distances
    """
    sys.path.insert(0, '/home/shrey/work/Phyla')
    os.chdir('/home/shrey/work/Phyla/phyla')
    from phyla.model.model import BiMambaWrapper
    
    def forward_with_residual(self, hidden_states, inference_params=None, cpu=None):
        out = self.mamba_fwd(hidden_states, inference_params=inference_params)
        if self.bidirectional:
            out_rev = self.mamba_rev(hidden_states.flip(dims=(1,)), 
                inference_params=inference_params).flip(dims=(1,))
            if self.bidirectional_strategy == "add":
                out = out + out_rev
            elif self.bidirectional_strategy == "ew_multiply":
                out = out * out_rev
        return hidden_states + out  # CRITICAL: Residual connection
    
    BiMambaWrapper.forward = forward_with_residual

def load_phyla_patched(device='cuda'):
    apply_residual_fix()
    from phyla.model.model import Phyla, Config
    model = Phyla(Config(), device=device)
    model.load()
    model.eval()
    return model
