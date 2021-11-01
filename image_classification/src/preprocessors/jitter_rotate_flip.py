from torchvision import transforms

preprocessor = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(25),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=15, translate=(0.1,0.1), scale=(0.8,0.8)),
    transforms.ToTensor(),
    transforms.Normalize((0.2861), (0.3528)) 
])