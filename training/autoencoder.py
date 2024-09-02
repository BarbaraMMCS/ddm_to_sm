from typing import List

from torch import nn, Tensor

class Reshape(nn.Module):
    def __init__(self, shape: List[int]):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.view(self.shape)

class Encoder(nn.Module):
    def __init__(self, name="Encoder"):
        self.name = name
        self.type = name
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(11*17, 6*8),
            nn.LeakyReLU(0.01),
            nn.Linear(6*8, 3*4),
            nn.LeakyReLU(0.01),
            nn.Linear(3*4, 2)
        )
    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


ENCODER = Encoder()

class Decoder(nn.Module):

    def __init__(self, name='Decoder'):
        super(Decoder, self).__init__()
        self.name = name
        self.type = name
        self.decoder = nn.Sequential(
                    nn.Linear(2, 3*4),
                    nn.LeakyReLU(0.01),
                    nn.Linear(3*4, 6*8),
                    nn.LeakyReLU(0.01),
                    nn.Linear(6*8, 11*17),
                    # nn.Sigmoid()
                )

    def forward(self, x: Tensor) -> Tensor:
        return self.decoder(x)


DECODER = Decoder()


class Autoencoder(nn.Module):

    def __init__(self, name='Autoencoder'):
        super(Autoencoder, self).__init__()
        self.name = name
        self.type = name
        self.encoder = ENCODER
        self.decoder = DECODER

    @property
    def n_params(self):
        return self.encoder.n_params + self.decoder.n_params

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        return self.decoder(x)


AUTOENCODER = Autoencoder()

class ConvEncoder(nn.Module):
    def __init__(self, name="ConvEncoder"):
        self.name = name
        self.type = name
        super(ConvEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, stride=(1, 1), kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, stride=(1, 1), kernel_size=(3, 3), padding=1),
            nn.Flatten(),
            nn.Linear(15 * 64, 2)
        )
    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


CONV_ENCODER = ConvEncoder()

class ConvDecoder(nn.Module):

    def __init__(self, name='ConvDecoder'):
        super(ConvDecoder, self).__init__()
        self.name = name
        self.type = name
        self.decoder = nn.Sequential(
                    nn.Linear(2, 15 * 64),
                    Reshape([-1, 64, 5, 3]),
                    nn.ConvTranspose2d(64, 64, stride=(1, 1), kernel_size=(3, 3), padding=1),
                    nn.LeakyReLU(0.01),
                    nn.ConvTranspose2d(64, 64, stride=(2, 2), kernel_size=(3, 3), padding=1, output_padding=(0, 1)),
                    nn.LeakyReLU(0.01),
                    nn.ConvTranspose2d(64, 32, stride=(2, 2), kernel_size=(3, 3), padding=1),
                    nn.LeakyReLU(0.01),
                    nn.ConvTranspose2d(32, 1, stride=(1, 1), kernel_size=(3, 3), padding=1),
                    # nn.Sigmoid()
                )

    def forward(self, x: Tensor) -> Tensor:
        return self.decoder(x)


CONV_DECODER = ConvDecoder()


class ConvAutoencoder(nn.Module):

    def __init__(self, name='ConvAutoencoder'):
        super(ConvAutoencoder, self).__init__()
        self.name = name
        self.type = name
        self.encoder = CONV_ENCODER
        self.decoder = CONV_DECODER

    @property
    def n_params(self):
        return self.encoder.n_params + self.decoder.n_params

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        return self.decoder(x)


CONV_AUTOENCODER = ConvAutoencoder()

if __name__ == '__main__':
    from torchsummary import summary

    AUTOENCODER.cuda()
    summary(AUTOENCODER, (1, 17, 11))

    # m = nn.Sequential(
    #     nn.Conv2d(1, 32, stride=(1, 1), kernel_size=(3, 3), padding=1),
    #     nn.LeakyReLU(0.01),
    #     nn.Conv2d(32, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
    #     nn.LeakyReLU(0.01),
    #     nn.Conv2d(64, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
    #     nn.LeakyReLU(0.01),
    #     nn.Conv2d(64, 64, stride=(1, 1), kernel_size=(3, 3), padding=1),
    #     nn.Flatten(),
    #     nn.Linear(15 * 64, 2),
    #     nn.Linear(2, 15 * 64),
    #     Reshape([-1, 64, 5, 3]),
    #     nn.ConvTranspose2d(64, 64, stride=(1, 1), kernel_size=(3, 3), padding=1),
    #     nn.LeakyReLU(0.01),
    #     nn.ConvTranspose2d(64, 64, stride=(2, 2), kernel_size=(3, 3), padding=1, output_padding=(0, 1)),
    #     nn.LeakyReLU(0.01),
    #     nn.ConvTranspose2d(64, 32, stride=(2, 2), kernel_size=(3, 3), padding=1),
    #     nn.LeakyReLU(0.01),
    #     nn.ConvTranspose2d(32, 1, stride=(1, 1), kernel_size=(3, 3), padding=1),
    #     nn.Sigmoid()
    # )
    # summary(m.cuda(), (1, 17, 11))

