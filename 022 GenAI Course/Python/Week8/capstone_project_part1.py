import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import data

# --- 1. Load a sample grayscale or colour image ---
image = data.astronaut()  
image_type = "Color"


# --- 2. Display the input image using matplotlib ---
print(f"Loaded a {image_type} image with shape: {image.shape} and data type: {image.dtype}")
plt.figure(figsize=(8, 8))
plt.imshow(image)
plt.title(f"Sample Image ({image_type})")
plt.axis('off')
# Show the plot
plt.show()

## Task 2: Simulate Forward Diffusion

x0 = image.astype(np.float32) / 255.0
T = 500 # Number of diffusion steps
beta_start = 0.0001
beta_end = 0.02
betas = np.linspace(beta_start, beta_end, T)
alphas = 1.0 - betas
alphas_cumprod = np.cumprod(alphas)

def forward_diffusion_step(x_prev, t, betas, alphas_cumprod):
    beta_t = betas[t - 1] 
    mean_part = np.sqrt(1.0 - beta_t) * x_prev
    noise = np.random.normal(size=x_prev.shape)
    std_part = np.sqrt(beta_t) * noise
    x_t = mean_part + std_part
    return x_t

steps_to_visualize = [0, 50, 150, 300, 499] 
images = [x0] # Start with the original image (t=0)
current_x = x0

# Run the diffusion process up to the last visualization step
for t in range(1, T + 1):
    current_x = forward_diffusion_step(current_x, t, betas, alphas_cumprod)
    
    if t in steps_to_visualize:
        images.append(current_x)

# --- Plotting the diffused images ---
fig, axes = plt.subplots(1, len(images), figsize=(16, 4))
fig.suptitle("Forward Diffusion Process (Adding Gaussian Noise)")

for i, x_t in enumerate(images):
    ax = axes[i]
    step_t = steps_to_visualize[i]
    
    # Clip and scale the image back to [0, 1]  for display
    display_image = np.clip(x_t, 0, 1) 
    
    ax.imshow(display_image)
    ax.set_title(f"Step t={step_t}")
    ax.axis('off')

plt.show()

print("\nTask2: Forward diffusion simulation complete.")