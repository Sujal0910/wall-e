import shutil
import os

print("1. Stitching PyTorch model back together...")

# Compress the folder back into a PyTorch-compatible architecture
shutil.make_archive("model_fixed", "zip", root_dir=".", base_dir="gpt2_wall_e_assistant")

print("2. Forcing the correct .pt extension...")
# Bypass Windows UI limitations to force the .pt extension
if os.path.exists("gpt2_wall_e_assistant.pt"):
    os.remove("gpt2_wall_e_assistant.pt")
os.rename("model_fixed.zip", "gpt2_wall_e_assistant.pt")

print("3. Done! The model is restored. You can now start the server.")