from torchvision import transforms

preprocessor = transforms.Compose([
    transforms.RandomVerticalFlip(p=0.6),
    transforms.RandomRotation(40),
    transforms.RandomHorizontalFlip(p=1),
    transforms.RandomAffine(degrees=30, translate=(0.1,0.1), scale=(0.8,0.8)),
    transforms.ToTensor(),
    transforms.Normalize((0.2861), (0.3528)) 
])