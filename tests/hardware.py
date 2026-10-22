"""Hardware capability checks for tests that need an Apple MPS device.

torch.backends.mps.is_available() only reports that the backend exists. GitHub's
hosted macOS arm64 runners report MPS as available, yet the first allocation fails
("MPS backend out of memory ... MPS allocated: 0 bytes"). The probe below therefore
allocates, computes and synchronizes. Only a RuntimeError marks the device unusable;
a wrong result is not treated as unavailability, so genuine numerical defects still
fail the MPS tests instead of being skipped.

Run as a script to print a hardware report (used by CI).
"""

import platform

import torch

SMALL = 1024  # elements; far below any real allocation limit


def mps_probe() -> tuple[bool, str]:
    """(usable, reason)."""
    if not torch.backends.mps.is_available():
        return False, "MPS backend not available on this machine"
    try:
        x = torch.ones(SMALL, device="mps")
        (x * 2).sum().item()
        torch.mps.synchronize()
    except RuntimeError as error:
        first_line = str(error).splitlines()[0]
        return False, f"MPS reported available but cannot allocate/compute: {first_line}"
    return True, "MPS allocation and computation succeeded"


MPS_USABLE, MPS_REASON = mps_probe()


if __name__ == "__main__":
    print(f"platform: {platform.platform()} ({platform.machine()})")
    print(f"python: {platform.python_version()}  torch: {torch.__version__}")
    print(f"torch.backends.mps.is_built(): {torch.backends.mps.is_built()}")
    print(f"torch.backends.mps.is_available(): {torch.backends.mps.is_available()}")
    print(f"MPS probe: {'usable' if MPS_USABLE else 'unusable'} ({MPS_REASON})")
