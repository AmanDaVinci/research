from torchvision import transforms

preprocessor = transforms.Compose([
    transforms.RandomRotation(degrees=(-90, 90)),
    transforms.RandomVerticalFlip(p=0.6),
    transforms.RandomRotation(40),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=10, translate=(0.1,0.1), scale=(0.8,0.8)),
    transforms.ToTensor(),
    transforms.Normalize((0.2861), (0.3528)) 
])