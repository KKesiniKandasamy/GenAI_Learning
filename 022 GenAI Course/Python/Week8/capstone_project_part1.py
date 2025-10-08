# TASK 1: Iport Libraries and Load a Sample Image

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import data

# --- 1. Load a sample grayscale or colour image ---
image = data.astronaut()  
image_type = "Color"


# --- 2. Display the input image using matplotlib ---
print(f"TASK1 Completed: Loaded a {image_type} image with shape: {image.shape} and data type: {image.dtype}")
plt.figure(figsize=(8, 8))
plt.imshow(image)
plt.title(f"Sample Image ({image_type})")
plt.axis('off')
plt.show()

## TASK 2: Simulate Forward Diffusion
x0 = image.astype(np.float32) / 255.0

# Define diffusion parameters
T = 1000  # Total number of diffusion steps 
beta_start = 0.0001
beta_end = 0.02
betas = np.linspace(beta_start, beta_end, T)

# Pre-calculate alpha & cumulative product of alphas
alphas = 1.0 - betas
alphas_cumprod = np.cumprod(alphas)


# --- Task: Create a function to add noise to an image based on a timestep t ---

def q_sample(x_start, t, alphas_cumprod):
    sqrt_bar_alpha_t = np.sqrt(alphas_cumprod[t - 1])
    one_minus_bar_alpha_t = 1.0 - alphas_cumprod[t - 1]
    sqrt_one_minus_bar_alpha_t = np.sqrt(one_minus_bar_alpha_t)
 
    noise = np.random.normal(size=x_start.shape)
    x_t = (sqrt_bar_alpha_t * x_start) + (sqrt_one_minus_bar_alpha_t * noise)
    
    return x_t


# --- Task: Apply the function and display the results ---

relative_timesteps = [0.1, 0.3, 0.6, 0.9]
# Calculate the corresponding integer steps 
integer_timesteps = [int(np.round(t_rel * T)) for t_rel in relative_timesteps]

# List to hold the noisy images
noisy_images = []

for t_int, t_rel in zip(integer_timesteps, relative_timesteps):
    t_safe = np.clip(t_int, 1, T) 
    
    print(f"Sampling at relative t={t_rel} (integer step t={t_safe})")
    x_t = q_sample(x0, t_safe, alphas_cumprod)
    
    # Store the image and its step label
    noisy_images.append({'image': x_t, 'label': f"t={t_rel} ({t_safe}/{T})"})

# --- Task: Organise visualisations in a subplot layout ---

num_images = len(noisy_images)
# Create a figure with one row and 'num_images' columns
fig, axes = plt.subplots(1, num_images, figsize=(4 * num_images, 5))
fig.suptitle("DDPM Forward Process: Direct Sampling $\mathbf{x}_t$ from $\mathbf{x}_0$", fontsize=14)

for i, data in enumerate(noisy_images):
    ax = axes[i]
    # Clip to [0, 1] range before displaying float images
    display_image = np.clip(data['image'], 0, 1) 
    ax.imshow(display_image)
    ax.set_title(data['label'])
    ax.axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout 
plt.show()


