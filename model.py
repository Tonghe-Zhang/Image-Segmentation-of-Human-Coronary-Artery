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
        )            #nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
    def forward(self, x):
        x = self.up(x)
        return x
"""
> Upsample: scale the HxW dimensions by 'scale_factor' times.
Fill the additional entries with biliner or nearest interpolation etc.
> Conv2d: 
"""



"""
        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

        x1,x2,x3,x4,x5=encoded_features



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


        print(f"x1.shape={x1.shape}")
        print(f"x2.shape={x2.shape}")
        print(f"x3.shape={x3.shape}")
        print(f"x4.shape={x4.shape}")
        print(f"x5.shape={x5.shape}")
        

        # U-net decoder
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)
        #print(f"d5.shape={d5.shape}")

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        #print(f"d4.shape={d4.shape}")

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        #print(f"d3.shape={d3.shape}")

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        #print(f"d2.shape={d2.shape}")

        d1 = self.Conv_1x1(d2)
        #print(f"d1.shape={d1.shape}")
"""



class UNetDecoderLayer(nn.Module):
    def __init__(self, ch_in:int, ch_out:int):
        super(UNetDecoderLayer, self).__init__()
        self.ch_in=ch_in
        self.ch_out=ch_out
        self.Up = up_conv(ch_in, ch_out)
        self.Conv = conv_block(ch_in, ch_out)
    def forward(self,x_link,decoded):
        print(f"x_link={x_link.shape}, d={decoded.shape}, ch_in={self.ch_in}, ch_out={self.ch_out}")
        decoded=self.Up(decoded)
        print(f"after up, decoded={decoded.shape}")
        decoded=torch.cat([x_link,decoded],dim=1)
        print(f"after cat, decoded={decoded.shape}")
        decoded=self.Conv(decoded)
        return decoded

class UNetDecoder(nn.Module):
    def __init__(self,
                 encoder_feat_ch=[1024, 512, 256, 128, 64],
                 decoder_feat_ch=[512,  256, 128, 64, 2]
                 ):
        """
        encoder_feat_ch: dimension of the channel of the encoder feature. Input order should be in decreasing order, x5, x4, x3, x2, x1
        """
        super(UNetDecoder, self).__init__()

        assert decoder_feat_ch[-1]==2


        self.output_W=512
        self.output_H=512

        self.num_layers=len(decoder_feat_ch)

        self.layer_Couts=decoder_feat_ch        #output channel dimension of each decoder layer.
        self.output_ch=decoder_feat_ch[-1]      #output segmentation image channel dimension, which is 2 by default (grey figure)

        self.layer_Cins=[_ for _ in range(self.num_layers)]   #input channel dimension of each decoder layer.
        #self.layer_Cins[0]=encoder_feat_ch[0]//2+encoder_feat_ch[1]
        for i in range(0,self.num_layers-1):
            #self.layer_Cins[i]=self.layer_Couts[i-1]//2+encoder_feat_ch[i+1]
            self.layer_Cins[i]=self.layer_Couts[i]+encoder_feat_ch[i+1]
            print(f"i={i}, self.layer_Couts[{i}]={self.layer_Couts[i]},\
                  encoder_feat_ch[{i+1}]={encoder_feat_ch[i+1]},\
                  self.layer_Cins[{i}]={self.layer_Cins[i]}")
        self.layer_Cins[-1]=self.layer_Couts[-2]

        
        print(self.layer_Cins)
        print(self.layer_Couts)
        """
        self.layer_Cins:  [1024, 512, 256, 128, 64]
        self.layer_Couts: [ 512, 256, 128,  64,  2]
        """

        self.decoder_layers=nn.ModuleList()
        for ch_in, ch_out in zip(self.layer_Cins, self.layer_Couts):
            self.decoder_layers.append(UNetDecoderLayer(ch_in=ch_in,ch_out=ch_out))
        
        output_conv=nn.Conv2d(in_channels=decoder_feat_ch[-2], out_channels=self.output_ch, kernel_size=1, stride=1, padding=0)
        self.decoder_layers.append(output_conv)

    def forward(self,encoded_features):
        # x5,x4,x3,x2,x1 = encoded_features
        decoded_features=[ _ for _ in range(self.num_layers) ]

        if encoded_features[0].shape[2:] != torch.Size([self.output_H//2**(self.num_layers-1), self.output_W//2**(self.num_layers-1)]):
            raise ValueError(f"Expected encoder output of shape [N, C, {self.output_H//2**(self.num_layers-1)}, {self.output_W//2**(self.num_layers-1)}], but got {encoded_features[0].shape} !")
            
        # d5, d4, d3, d2, d1
        """
        Decoder process, equivalent to 
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Conv5(d5)                 C4+C5 -> 

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)     
        d4 = self.Conv4(d4)      

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Conv2(d2)

        d1 = self.Conv_1x1(d2)

        when there are five layers. 
        """

        d=encoded_features[0]
        decoded_features[0]=self.decoder_layers[0](x_link=encoded_features[1],decoded=encoded_features[0])
        for feat in range(1,self.num_layers-1):
            d=decoded_features[feat-1]
            x=encoded_features[feat+1]
            
            d=self.decoder_layers[feat](x_link=x,decoded=d)
            decoded_features[feat]=d
        d=self.decoder_layers[-1](decoded_features[-2])
        decoded_features[-1]=d

        """
        Current Architecture:
        H and W:
            after passing through each layer before the last layaer, H and W shrinks to 1/2. The last layer does not alter H and W. 
        C:
            channel changes according to "self.decoder_layer_ch"
        
        U-net also asks the H and W of x_{i-1} should be 2 times of that of x_{i}, while it does not restrict the relationship between x_i and x_{i+1}'s channels. 
        However the encoder's channels x_5...x_1 should be the same as the decoder's "self.decoder_layer_ch", while the last channel of the decoder(output channel) 
        is determined by the output image's configurations. (in our setting the grey image has channel 2.)
        """

        """
        shape information:

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
        
        for i in decoded_features:
             print(i.shape)
        # for i in decoded_features:
        #      print(f"d[{num_encodes-i+1}].shape={i.shape}")
        """
        for i, feat in enumerate(encoded_features):
             print(f"encoded{i} ={feat.shape}")
        for i, feat in enumerate(decoded_features):
             print(f"decoded{i} ={feat.shape}")

        if d.shape[1:] != torch.Size([2,self.output_H,self.output_W]):
            raise ValueError(f"Expected decoder output of shape [N,2,{self.output_H},{self.output_W}], but got {d.shape} !")
        return d

minibathsize=4
output_H=512
output_W=512

encode_ch=[2048,1024,512,256,128]   #[1024, 512, 256, 128, 64]
num_encodes=5

input_H=output_H
intpu_W=output_W
encoded_features=[_ for _ in range(num_encodes)]
for i in range(num_encodes):
    encoded_features[i]=torch.randn(minibathsize,encode_ch[i],input_H//(2**(num_encodes-1-i)),intpu_W//(2**(num_encodes-1-i)))

for i in range(num_encodes):
    print(encoded_features[i].shape)

decoder=UNetDecoder(encoder_feat_ch=encode_ch, decoder_feat_ch=[512, 256, 128, 64, 2])
d=decoder(encoded_features)

"""
x1=torch.randn(4,64,512,512)
x2=torch.randn(4,128,256,256)
x3=torch.randn(4,256,128,128)
x4=torch.randn(4,512,64,64)
x5=torch.randn(4,1024,32,32)
encoded_features=[x5,x4,x3,x2,x1]
encoder_channels=[1024,512,512,128,64]
"""