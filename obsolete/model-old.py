
class ResUNet_old(nn.Module):
    def __init__(self, num_classes=2):
        super(ResUNet_old, self).__init__()

        # Builing the encoder from resenet 50
        self.model = resnet50(pretrained=True)  # Load pre-trained ResNet-50        
        # freeze the pretrained encoder. only allow training of the U net decoder layers. 
        for param in self.model.parameters():
            param.requires_grad = False

        # create learnable layer 1:
        # input dimension: [N=4, 1, H=512, W=512], output dimension: [N=4, 64, H=512, W=512]
        self.conv1=nn.Conv2d(1, 64, kernel_size=3, stride=1,padding=1, bias=True)
        self.bn1=self.model.bn1
        self.maxpool1=nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
        self.layer1=torch.nn.Sequential(self.conv1,self.bn1, self.maxpool1)

        #create learnable layer 2:
        # input dimension: [N=4, 64, H=512, W=512], output dimension: [N=4, 128, H=256, H=256]
        self.conv2=nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2=nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.maxpool2=nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.layer2=nn.Sequential(self.conv2, self.bn2, self.maxpool2)
        
        # layer 3: input dimension: [N=4, 128, H=256, H=256],
        # output dimension: [4,256,128,128]
        self.conv3=nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3=nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.maxpool3=nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.layer3=nn.Sequential(self.conv3, self.bn3, self.maxpool3)

        #borrow other layers from resnet50
        # layer 4: input dimension: [4,256,128,128], output dimension: [4, 512, 64, 64]
        self.layer4=self.model.layer2
        # layer 5: input dimension: [4, 512, 64, 64], output dimension: [4, 1024, 32, 32]
        self.layer5=self.model.layer3
                
        # use the Unet decoder as the decoder
        self.decoder = UNetDecoder(num_classes)
        
    def forward(self, x):
        # Encoder layers forward pass
        x1=self.layer1(x)     #x1: [4, 64, 512, 512]
        x2=self.layer2(x1)    #x2: [4, 128, 256, 256]
        x3=self.layer3(x2)    #x3: [4,256,128,128]     
        x4=self.layer4(x3)	 #x4: [4, 512, 64, 64]
        x5=self.layer5(x4)    #x5: [4, 1024, 32, 32]
        encoder_embeddings=[x1,x2,x3,x4,x5]
        """
        print(f"x.shape={x.shape}")
        print(f"x1.shape={x1.shape}")
        print(f"x2.shape={x2.shape}")
        print(f"x3.shape={x3.shape}")
        print(f"x4.shape={x4.shape}")
        print(f"x5.shape={x5.shape}")
        x.shape=torch.Size([4, 1, 512, 512])
        x1.shape=torch.Size([4, 64, 128, 128])
        x2.shape=torch.Size([4, 256, 128, 128])
        x3.shape=torch.Size([4, 512, 64, 64])
        x4.shape=torch.Size([4, 1024, 32, 32])
        x5.shape=torch.Size([4, 2048, 16, 16])
        exit()
        """
        # Decoder
        d1 = self.decoder(encoder_embeddings)
        #print(f"d1.shape={d1.shape}")
        return d1
