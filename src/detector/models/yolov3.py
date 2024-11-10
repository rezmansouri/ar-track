import torch
import torch.nn as nn

class CNNBlock(nn.Module):
	def __init__(self, in_channels, out_channels, use_batch_norm=True, **kwargs):
		super().__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, bias=not use_batch_norm, **kwargs)
		self.bn = nn.BatchNorm2d(out_channels)
		self.activation = nn.LeakyReLU(0.1)
		self.use_batch_norm = use_batch_norm
	def forward(self, x):
		x = self.conv(x)
		if self.use_batch_norm:
			x = self.bn(x)
			return self.activation(x)
		else:
			return x

class ResidualBlock(nn.Module):
	def __init__(self, channels, use_residual=True, num_repeats=1):
		super().__init__()
		
		
		res_layers = []
		for _ in range(num_repeats):
			res_layers += [
				nn.Sequential(
					nn.Conv2d(channels, channels // 2, kernel_size=1),
					nn.BatchNorm2d(channels // 2),
					nn.LeakyReLU(0.1),
					nn.Conv2d(channels // 2, channels, kernel_size=3, padding=1),
					nn.BatchNorm2d(channels),
					nn.LeakyReLU(0.1)
				)
			]
		self.layers = nn.ModuleList(res_layers)
		self.use_residual = use_residual
		self.num_repeats = num_repeats
	
	
	def forward(self, x):
		for layer in self.layers:
			residual = x
			x = layer(x)
			if self.use_residual:
				x = x + residual
		return x

class ScalePrediction(nn.Module):
	def __init__(self, in_channels):
		super().__init__()
		
		self.pred = nn.Sequential(
			nn.Conv2d(in_channels, 2*in_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(2*in_channels),
			nn.LeakyReLU(0.1),
			nn.Conv2d(2*in_channels, 5 * 3, kernel_size=1),
		)
	
	
	
	def forward(self, x):
		output = self.pred(x)
		output = output.view(x.size(0), 3, 5, x.size(2), x.size(3))
		output = output.permute(0, 1, 3, 4, 2)
		return output

class YOLOv3(nn.Module):
	def __init__(self, in_channels=1):
		super().__init__()
		self.in_channels = in_channels
		
		self.layers = nn.ModuleList([
			CNNBlock(in_channels, 32, kernel_size=3, stride=1, padding=1),
			CNNBlock(32, 64, kernel_size=3, stride=2, padding=1),
			ResidualBlock(64, num_repeats=1),
			CNNBlock(64, 128, kernel_size=3, stride=2, padding=1),
			ResidualBlock(128, num_repeats=2),
			CNNBlock(128, 256, kernel_size=3, stride=2, padding=1),
			ResidualBlock(256, num_repeats=8),
			CNNBlock(256, 512, kernel_size=3, stride=2, padding=1),
			ResidualBlock(512, num_repeats=8),
			CNNBlock(512, 1024, kernel_size=3, stride=2, padding=1),
			ResidualBlock(1024, num_repeats=4),
			CNNBlock(1024, 512, kernel_size=1, stride=1, padding=0),
			CNNBlock(512, 1024, kernel_size=3, stride=1, padding=1),
			ResidualBlock(1024, use_residual=False, num_repeats=1),
			CNNBlock(1024, 512, kernel_size=1, stride=1, padding=0),
			ScalePrediction(512),
			CNNBlock(512, 256, kernel_size=1, stride=1, padding=0),
			nn.Upsample(scale_factor=2),
			CNNBlock(768, 256, kernel_size=1, stride=1, padding=0),
			CNNBlock(256, 512, kernel_size=3, stride=1, padding=1),
			ResidualBlock(512, use_residual=False, num_repeats=1),
			CNNBlock(512, 256, kernel_size=1, stride=1, padding=0),
			ScalePrediction(256),
			CNNBlock(256, 128, kernel_size=1, stride=1, padding=0),
			nn.Upsample(scale_factor=2),
			CNNBlock(384, 128, kernel_size=1, stride=1, padding=0),
			CNNBlock(128, 256, kernel_size=3, stride=1, padding=1),
			ResidualBlock(256, use_residual=False, num_repeats=1),
			CNNBlock(256, 128, kernel_size=1, stride=1, padding=0),
			ScalePrediction(128)
		])
	
	def forward(self, x):
		outputs = []
		route_connections = []
		for layer in self.layers:
			if isinstance(layer, ScalePrediction):
				outputs.append(layer(x))
				continue
			x = layer(x)
			if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
				route_connections.append(x)
			elif isinstance(layer, nn.Upsample):
				x = torch.cat([x, route_connections[-1]], dim=1)
				route_connections.pop()
		return outputs
