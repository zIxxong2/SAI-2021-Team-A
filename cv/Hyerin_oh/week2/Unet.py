import torch
import torch.nn as nn

class unet(nn.Module):
  
    def __init__(self,num_classes):
        super(unet, self).__init__()
        # input_shape : [B,3,572,572]
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 0)
        self.conv2 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding = 0)
        self.downsample1 = self.down_block(64,128)      # [b,64,284,284]
        self.downsample2 = self.down_block(128,256)    # [b,128,140,140]
        self.downsample3 = self.down_block(256,512)   # [b,256,68,68]
        self.downsample4 = self.down_block(512,1024)   # [b,512,32]

        self.upsample4_0 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride = 2, padding = 0)
        self.upsample4_1 = self.up_block(1024,512)
        self.upsample3_0 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride = 2, padding = 0)
        self.upsample3_1 = self.up_block(512,256)
        self.upsample2_0 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride = 2, padding = 0)
        self.upsample2_1 = self.up_block(256,128)
        self.upsample1_0 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride = 2, padding = 0)
        self.upsample1_1 = self.up_block(128,64)

        self.score = nn.Conv2d(64,num_classes,kernel_size=1)

    def forward(self,x) :
        x = self.conv1(x)
        x = self.conv2(x)

        pool1 = x 
        x = self.downsample1(x)

        pool2 = x
        x = self.downsample2(x)

        pool3 = x
        x = self.downsample3(x)

        pool4 = x
        x = self.downsample4(x)

        x = torch.cat((self.upsample4_0(x) , self.crop_image(pool4 , 64, 56)) , dim = 1)
        x = self.upsample4_1(x)

        x = torch.cat((self.upsample3_0(x) , self.crop_image(pool3 , 136, 104)) , dim = 1)
        x = self.upsample3_1(x)

        x = torch.cat((self.upsample2_0(x) , self.crop_image(pool2 , 280, 200)) , dim = 1)
        x = self.upsample2_1(x)

        x = torch.cat((self.upsample1_0(x) , self.crop_image(pool1 , 568, 392)) , dim = 1)
        x = self.upsample1_1(x)

        x = self.score(x)

        return x 


    def down_block(self, in_channels, out_channels):
        layers = [nn.MaxPool2d(kernel_size = 2, stride = 2),
                  nn.Conv2d(in_channels, out_channels , kernel_size = 3, padding = 0 , stride = 1),
                  nn.BatchNorm2d(out_channels),
                  nn.ReLU(True),  
                  nn.Conv2d(out_channels , out_channels, kernel_size = 3, padding = 0, stride = 1),
                  nn.BatchNorm2d(out_channels),
                  nn.ReLU(True),  
                  ]
        return nn.Sequential(*layers)


    def up_block(self, in_channels, out_channels):
        layers = [
                  nn.Conv2d(in_channels, out_channels , kernel_size = 3, padding = 0 , stride = 1),
                  nn.BatchNorm2d(out_channels),
                  nn.ReLU(True),  
                  nn.Conv2d(out_channels , out_channels, kernel_size = 3, padding = 0, stride = 1),
                  nn.BatchNorm2d(out_channels),
                  nn.ReLU(True),  
                  ]
        return nn.Sequential(*layers)


    def crop_image(self, x, output_size, input_size):
        return x[ : , 
                 : ,  
                 int((output_size - input_size) / 2 ): int((output_size + input_size) / 2) , 
                 int((output_size - input_size) / 2 ): int((output_size + input_size) / 2) ]


def UNET(num_class):
    return unet(num_class) 
