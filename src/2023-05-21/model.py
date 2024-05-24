import torch
from torch import nn
from torchvision.models import resnet50


class UNet_vanilla(nn.Module):
    def __init__(self, img_ch=1, output_ch=2):
        """
        img_ch=1 because input image is a grey image, whose feature only contains one channel. 
        > Input is of shape torch.Size([4,1,512,512]), 
        where 4 is the batch size, 1 is the channel number, 512, 512 are the width and height.
        output_ch=2 because we output a black/white distribution. 
        > Output is of shape torch.Size([4, 2, 512, 512]). 
        """
        super().__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)
        
        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # U-net encoder
        x1 = self.Conv1(x)
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        
        # U-net decoder
        d5 = self.Up5(x5)# input [4, 1024, 32, 32], output [4, 512, 64, 64]
        d5 = torch.cat((x4, d5), dim=1) # input [4, 512, 64, 64] and [4, 512, 64, 64], output [4, 1024, 64, 64]
        d5 = self.Up_conv5(d5) # input  [4, 1024, 64, 64], output [4, 512, 64, 64]
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        
        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)
        
        """
        print(f"x.shape={x.shape}")
        print(f"x1.shape={x1.shape}")
        print(f"x2.shape={x2.shape}")
        print(f"x3.shape={x3.shape}")
        print(f"x4.shape={x4.shape}")
        print(f"x5.shape={x5.shape}")
        
        print(f"d5.shape={d5.shape}")
        print(f"d4.shape={d4.shape}")
        print(f"d3.shape={d3.shape}")
        print(f"d2.shape={d2.shape}")
        print(f"d1.shape={d1.shape}")
        exit()
        """
        """
        x.shape=torch.Size([4, 1 512, 512])

        x1.shape=torch.Size([4, 64, 512, 512])
        x2.shape=torch.Size([4, 128, 256, 256])
        x3.shape=torch.Size([4, 256, 128, 128])
        x4.shape=torch.Size([4, 512, 64, 64])
        x5.shape=torch.Size([4, 1024, 32, 32])
        
        d5.shape=torch.Size([4, 512, 64, 64])
        d4.shape=torch.Size([4, 256, 128, 128])
        d3.shape=torch.Size([4, 128, 256, 256])
        d2.shape=torch.Size([4, 64, 512, 512])
        d1.shape=torch.Size([4, 2, 512, 512])
        
        output channel feature dimension is of "2",
        because it is a black/white distribution(binary distribution). "4" is the batch size, "512, 512" are the width and height.
        """
        return d1

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.up(x)
        return x
    

class UNetDecoder(nn.Module):
    def __init__(self,output_ch:int):
        super(UNetDecoder, self).__init__()
        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)
        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)
        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self,encoded_features):
        x1,x2,x3,x4,x5=encoded_features
        """ 
        
        """
        # U-net decoder
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        
        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        
        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        
class ResUNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ResUNet, self).__init__()

        # Builing the encoder from resenet 50
        self.encoder = resnet50(pretrained=True)  # Load pre-trained ResNet-50
        # [new stuff]
        # Since our dataset inputs gray images while resnet50 is trained on the RGB imagenet dataset, we =
        # need to revise the first layer of resnet 50
        self.encoder.conv1=nn. Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.layer0 = nn.Sequential(self.encoder.conv1, self.encoder.bn1, self.encoder.relu, self.encoder.maxpool)
        self.layer1 = self.encoder.layer1  # (64x64 levels)
        self.layer2 = self.encoder.layer2  # (128x128 levels)
        self.layer3 = self.encoder.layer3  # (256x256 levels)
        self.layer4 = self.encoder.layer4  # (2048x2048 levels)
        # freeze the pretrained encoder. only allow training of the U net decoder layers. 
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # use the Unet decoder as the decoder
        self.decoder = UNetDecoder(num_classes)
        
    def forward(self, x):
        # Encoder
        x1 = self.layer0(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        
        features = [x1, x2, x3, x4, x5]
        
        print(f"x.shape={x.shape}")
        print(f"x1.shape={x1.shape}")
        print(f"x2.shape={x2.shape}")
        print(f"x3.shape={x3.shape}")
        print(f"x4.shape={x4.shape}")
        print(f"x5.shape={x5.shape}")
        """
        x.shape=torch.Size([4, 1, 512, 512])
        x1.shape=torch.Size([4, 64, 128, 128])
        x2.shape=torch.Size([4, 256, 128, 128])
        x3.shape=torch.Size([4, 512, 64, 64])
        x4.shape=torch.Size([4, 1024, 32, 32])
        x5.shape=torch.Size([4, 2048, 16, 16])
        """
        exit()
        # Decoder
        d1 = self.decoder(features)
        
        return d1
