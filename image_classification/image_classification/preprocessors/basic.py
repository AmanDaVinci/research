from torchvision import transforms

preprocessor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2869), (0.3540)) 
])