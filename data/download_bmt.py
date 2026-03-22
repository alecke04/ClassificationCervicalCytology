"""
Download BMT dataset from Synapse
"""

import os
import sys
from getpass import getpass
from pathlib import Path

try:
    import synapseclient
    from synapseutils import syncFromSynapse
    from synapseclient.core.credentials.credential_provider import get_config_authentication
except ImportError:
    print("Error: synapseclient is not installed")
    print("Install with: pip install synapseclient")
    sys.exit(1)


def download_bmt_dataset(
    output_dir: str = "data/raw",
    username: str = None,
    auth_token: str = None,
):
    """
    Download BMT dataset from Synapse.
    
    Args:
        output_dir: Output directory to save images
        username: Synapse username (optional)
        auth_token: Synapse Personal Access Token (optional)
    
    Dataset: syn55259257 (BMT - Balanced Multi-cell ThinPrep)
    Reference: https://www.synapse.org/Synapse:syn55259257/wiki/629294
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _organize_bmt_files(base_dir: Path) -> None:
        """Move flat downloaded files into NILM/LSIL/HSIL folders based on filename prefix."""
        source_dir = base_dir / "Brown Multicellular ThinPrep Database - images"
        if not source_dir.exists():
            return

        class_dirs = {
            "NILM": base_dir / "NILM",
            "LSIL": base_dir / "LSIL",
            "HSIL": base_dir / "HSIL",
        }
        for class_dir in class_dirs.values():
            class_dir.mkdir(parents=True, exist_ok=True)

        moved = 0
        for file_path in source_dir.glob("*"):
            if not file_path.is_file():
                continue

            name = file_path.name.lower()
            if name.startswith("hsil"):
                target_dir = class_dirs["HSIL"]
            elif name.startswith("lsil"):
                target_dir = class_dirs["LSIL"]
            elif name.startswith("nilm") or name.startswith("nil"):
                target_dir = class_dirs["NILM"]
            else:
                # Keep manifest/unknown files in place.
                continue

            target_path = target_dir / file_path.name
            if file_path.resolve() != target_path.resolve():
                file_path.replace(target_path)
                moved += 1

        if moved:
            print(f"✓ Organized {moved} images into NILM/LSIL/HSIL directories")
    
    print(f"\n{'='*60}")
    print("Downloading BMT Dataset from Synapse")
    print(f"{'='*60}\n")
    
    try:
        # Initialize Synapse client
        syn = synapseclient.Synapse()

        # Inspect auth config used by Synapse client.
        config_auth = get_config_authentication(config_path=syn.configPath)
        
        # Login priority:
        # 1) CLI arg --auth_token
        # 2) Environment variable SYNAPSE_AUTH_TOKEN
        # 3) ~/.synapseConfig
        token = auth_token or os.getenv("SYNAPSE_AUTH_TOKEN")
        config_token = config_auth.get("authtoken")
        if token:
            print("Logging in with provided Synapse Auth Token...")
            syn.login(email=username, authToken=token, silent=False)
        elif config_token:
            print("Logging in using token from ~/.synapseConfig...")
            syn.login(email=username)
        else:
            print("No token found in arguments, environment, or ~/.synapseConfig.")
            prompted_token = getpass("Paste Synapse Personal Access Token (input hidden): ").strip()
            if not prompted_token:
                raise RuntimeError("No auth token entered.")
            print("Logging in with prompted Synapse Auth Token...")
            # Avoid username mismatch edge-cases by authenticating with token directly.
            syn.login(authToken=prompted_token, silent=False)
        
        # Download dataset
        print(f"\nDownloading to: {output_dir}")
        print("This may take 10-30 minutes depending on your connection...")
        
        # synapseutils API changed arg names across versions.
        # Try modern 'path' first, then fallback to legacy 'downloadLocation'.
        try:
            syncFromSynapse(
                syn,
                'syn55259257',
                path=str(output_dir),
            )
        except TypeError:
            syncFromSynapse(
                syn,
                'syn55259257',
                downloadLocation=str(output_dir),
            )

            # Ensure downstream training sees expected structure: data/raw/NILM|LSIL|HSIL
            _organize_bmt_files(output_dir)
        
        print(f"\n✓ Download complete!")
        print(f"✓ Dataset saved to {output_dir}")
        
        # Verify structure
        print(f"\nDataset structure:")
        for class_dir in output_dir.iterdir():
            if class_dir.is_dir():
                num_images = len(list(class_dir.glob("*.jpg"))) + len(list(class_dir.glob("*.png")))
                print(f"  {class_dir.name}: {num_images} images")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure you have a Synapse account: https://www.synapse.org/")
        print("2. Install required package: pip install synapseclient")
        print("3. Create token in Synapse: User Profile -> Settings -> Personal Access Tokens")
        print("4. Set env var for this shell: $env:SYNAPSE_AUTH_TOKEN='<token>'")
        print("5. Or store token in C:\\Users\\<you>\\.synapseConfig:")
        print("   [authentication]")
        print("   username = <your_username>")
        print("   authtoken = <your_token>")
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download BMT dataset from Synapse")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/raw",
        help="Output directory (default: data/raw)"
    )
    parser.add_argument(
        "--username",
        type=str,
        default=None,
        help="Synapse username (optional)"
    )
    parser.add_argument(
        "--auth_token",
        type=str,
        default=None,
        help="Synapse Personal Access Token (optional; can also use SYNAPSE_AUTH_TOKEN env var)"
    )
    
    args = parser.parse_args()
    download_bmt_dataset(
        output_dir=args.output_dir,
        username=args.username,
        auth_token=args.auth_token,
    )
