import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# 定义生成器
class Generator(nn.Module):
    def __init__(self, latent_dim, label_dim, embed_dim, img_shape):
        super(Generator, self).__init__()
        self.label_embedding = nn.Embedding(label_dim, embed_dim)
        
        self.model = nn.Sequential(
            # 输入是 latent_dim + embed_dim 维的噪声和标签嵌入
            nn.ConvTranspose2d(latent_dim + embed_dim, 512, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, img_shape[0], kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        self.img_shape = img_shape

    def forward(self, noise, labels):
        # 嵌入标签
        label_embed = self.label_embedding(labels)
        # 拼接噪声和标签嵌入
        gen_input = torch.cat((noise, label_embed), dim=1)
        # 调整形状以适应卷积层
        gen_input = gen_input.unsqueeze(2).unsqueeze(3)
        # 生成图像
        img = self.model(gen_input)
        return img

# 定义判别器
class Discriminator(nn.Module):
    def __init__(self, label_dim, embed_dim, img_shape):
        super(Discriminator, self).__init__()
        self.label_embedding = nn.Embedding(label_dim, embed_dim)
        
        self.model = nn.Sequential(
            nn.Conv2d(img_shape[0] + 1, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        # 嵌入标签
        label_embed = self.label_embedding(labels)
        # 调整标签嵌入的形状以匹配图像的形状
        label_embed = label_embed.view(label_embed.size(0), label_embed.size(1), 1, 1)
        label_embed = label_embed.expand(label_embed.size(0), label_embed.size(1), img.size(2), img.size(3))
        # 拼接图像和标签嵌入
        d_in = torch.cat((img, label_embed), dim=1)
        # 判别图像真伪
        validity = self.model(d_in)
        return validity

# 超参数
latent_dim = 100
label_dim = 10  # 假设有10个类别
embed_dim = 10
img_shape = (1, 28, 28)  # MNIST图像尺寸
batch_size = 64
lr = 0.0002
n_epochs = 200

# 初始化生成器和判别器
generator = Generator(latent_dim, label_dim, embed_dim, img_shape)
discriminator = Discriminator(label_dim, embed_dim, img_shape)

# 优化器
optimizer_G = optim.Adam(generator.parameters(), lr=lr)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

# 损失函数
adversarial_loss = nn.BCELoss()

# 数据加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 初始化列表来存储损失值
d_losses = []
g_losses = []

# 训练循环
for epoch in range(n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):
        # 真实样本
        real_imgs = imgs
        real_labels = labels
        # 生成噪声和假标签
        z = torch.randn(imgs.size(0), latent_dim)
        fake_labels = torch.randint(0, label_dim, (imgs.size(0),))
        
        # 生成假样本
        fake_imgs = generator(z, fake_labels)
        
        # 训练判别器
        optimizer_D.zero_grad()
        real_loss = adversarial_loss(discriminator(real_imgs, real_labels), torch.ones(imgs.size(0), 1))
        fake_loss = adversarial_loss(discriminator(fake_imgs.detach(), fake_labels), torch.zeros(imgs.size(0), 1))
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()
        
        # 训练生成器
        optimizer_G.zero_grad()
        g_loss = adversarial_loss(discriminator(fake_imgs, fake_labels), torch.ones(imgs.size(0), 1))
        g_loss.backward()
        optimizer_G.step()
        
        # 记录损失值
        d_losses.append(d_loss.item())
        g_losses.append(g_loss.item())
        
        if i % 100 == 0:
            print(f"[Epoch {epoch}/{n_epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

# 绘制损失曲线
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(d_losses, label='Discriminator Loss')
plt.plot(g_losses, label='Generator Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training Losses')
plt.legend()
plt.show()

# 生成1-9的数字图片
def generate_images(generator, latent_dim, label_dim, n_images=9):
    generator.eval()
    with torch.no_grad():
        # 生成噪声
        z = torch.randn(n_images, latent_dim)
        # 生成标签（1-9）
        labels = torch.arange(0, n_images).long()
        # 生成图片
        gen_imgs = generator(z, labels)
        # 将图片从[-1, 1]转换到[0, 1]
        gen_imgs = 0.5 * gen_imgs + 0.5
        return gen_imgs, labels

# 生成并显示图片
gen_imgs, labels = generate_images(generator, latent_dim, label_dim)

# 显示生成的图片
fig, axes = plt.subplots(3, 3, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    ax.imshow(gen_imgs[i].squeeze().numpy(), cmap='gray')
    ax.set_title(f'Label: {labels[i].item()}')
    ax.axis('off')
plt.show()