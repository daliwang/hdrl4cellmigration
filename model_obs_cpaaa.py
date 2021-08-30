from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from sklearn import svm
import _pickle as pickle
import random
import sys, time, argparse
import numpy as np
import operator

import io
from PIL import Image, ImageDraw

from draw_plane import DrawPlane
from embryo import Embryo
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as utils



parser = argparse.ArgumentParser()
parser.add_argument("--emb", type=int, default=3, help="The index of Cpaaa embryo. choose from [0,1,2,3]")

args = parser.parse_args()
if args.emb == 0:
	START_POINT = 168
	END_POINT = 190
	PLANE_DRAW = 7
	NUCLEI_DATA_PATH = './data/cpaaa_0/'	
elif args.emb == 1:
	START_POINT = 100
	END_POINT = 125
	PLANE_DRAW = 7
	NUCLEI_DATA_PATH = './data/cpaaa_1/'
elif args.emb == 2:
	START_POINT = 113
	END_POINT = 137-4
	PLANE_DRAW = 7
	NUCLEI_DATA_PATH = './data/cpaaa_2/'
elif args.emb == 3:
	START_POINT = 104
	END_POINT = 135-7
	PLANE_DRAW = 7
	NUCLEI_DATA_PATH = './data/cpaaa_3/'
elif args.emb == 4:
	START_POINT = 100
	END_POINT = 133-9
	PLANE_DRAW = 7
	NUCLEI_DATA_PATH = './data/cpaaa_4/'
else:
	print('Error. Invalid embryo!')
	exit()

TICK_RESOLUTION = 10
AI_CELL = 'Cpaaa'


PLANE_THRESHOLD = 3			#draw the cells in the [-2,+2] planes (5 total)


INPUT_PLANE_RANGE = 2 #0

CANVAS_DISPLAY_SCALE_FACTOR = 2		### zoom in/out the size of canvas
FRESH_TIME = 0.02
FRESH_PERIOD = 1

PLANE_RESOLUTION = 4


import torch._utils
try:
	torch._utils._rebuild_tensor_v2
except AttributeError:
	def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
		tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
		tensor.requires_grad = requires_grad
		tensor._backward_hooks = backward_hooks
		return tensor
	torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

class Net(nn.Module):
	def __init__(self, n_input):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(n_input, 20480)
		self.fc2 = nn.Linear(20480, 1024)
		self.fc3 = nn.Linear(1024, 256)
		self.out = nn.Linear(256, 2)

	def forward(self, x):
		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		x = F.relu(x)
		x = self.fc3(x)
		x = F.relu(x)
		result = self.out(x)
		return result

class SeqRosModel(Model):
	def __init__(self):
		self.speed_model_plus = Net(4096)
		self.speed_model_plus.load_state_dict(torch.load('./trained_models/TMM.pkl', map_location=lambda storage, loc: storage))
		self.file_path = NUCLEI_DATA_PATH + 't%03d-nuclei'

		print('Parsing the Embryo...')
		self.embryo = Embryo(NUCLEI_DATA_PATH)
		self.embryo.read_data()
		self.embryo.get_embryo_visual_params()

		self.embryo.volume = 2500578

		self.ai_cell = AI_CELL

		self.start_point = START_POINT
		self.end_point = END_POINT

		self.ticks = 0
		self.tick_resolution = TICK_RESOLUTION
		self.end_tick = (self.end_point - self.start_point) * self.tick_resolution
		self.stage_destination_point = self.start_point

		self.plane_resolution = PLANE_RESOLUTION


		self.current_cell_list = []
		self.dividing_cell_overall = []
		self.next_stage_destination_list = {}
		self.schedule = RandomActivation(self)

		self.init_env()
		self.update_stage_destination()

		self.plane = DrawPlane(width=self.embryo.width,
							height=self.embryo.height,
							w_offset = self.embryo.wid_offset,
							h_offset = self.embryo.hei_offset,
							scale_factor = CANVAS_DISPLAY_SCALE_FACTOR)
		self.canvas = self.plane.canvas
		self.plane_draw = PLANE_DRAW
		self.draw(self.plane_draw)


	def draw(self,n_plane):
		self.canvas.delete("all")
		draw_range = np.arange(n_plane-PLANE_THRESHOLD, n_plane+PLANE_THRESHOLD+1, 1)
		draw_range = draw_range.tolist()
		draw_range.reverse()

		for n in draw_range:
			angle = np.pi *0.5 / (PLANE_THRESHOLD + 1) * np.abs(n - n_plane)
			level = None
			for cell in self.schedule.agents:
				if cell.cell_name == self.ai_cell:
					type = 'AI'
				else:
					type = 'NUMB'
				if round(cell.location[2]) == n:
					self.plane.draw_cell(center=cell.location[0:2],
										radius=cell.diameter/2.0*np.cos(angle),
										type=type,
										level=level)
		self.canvas.pack()
		self.canvas.update()
		time.sleep(FRESH_TIME)

	def radis_ratio(self, cn):
		r=-1
		if cn[0:2]=="AB":
			r=0.55*(0.5**(len(cn)-2))
		elif cn=="P1":
			r=0.45
		elif cn=="EMS":
			r=0.45*0.54
		elif cn=="P2":
			r=0.45*0.46
		elif cn[0:2]=="MS":
			r=0.45*0.54*0.5*(0.5**(len(cn)-2))
		elif cn=="E":
			r=0.45*0.54*0.5
		elif cn[0]=="E" and len(cn)>=2 and cn[1] != "M":
			r=0.45*0.54*0.5*(0.5**(len(cn)-1))
		elif cn[0]=="C":
			r=0.45*0.46*0.53*(0.5**(len(cn)-1))
		elif cn=="P3":
			r=0.45*0.46*0.47
		elif cn[0]=="D":
			r=0.45*0.46*0.47*0.52*(0.5**(len(cn)-1))
		elif cn=="P4":
			r=0.45*0.46*0.47*0.48
		if r == -1:
			return 0.00000001
		return r**(1.0/3)

	def get_radius(self, cell_name):
		if cell_name[0:2]=="AB":
			v=0.55*(0.5**(len(cell_name)-2))
		elif cell_name=="P1":
			v=0.45
		elif cell_name=="EMS":
			v=0.45*0.54
		elif cell_name=="P2":
			v=0.45*0.46
		elif cell_name[0:2]=="MS":
			v=0.45*0.54*0.5*(0.5**(len(cell_name)-2))
		elif cell_name=="E":
			v=0.45*0.54*0.5
		elif cell_name[0]=="E" and len(cell_name)>=2 and cell_name[1] != "M":
			v=0.45*0.54*0.5*(0.5**(len(cell_name)-1))
		elif cell_name[0]=="C":
			v=0.45*0.46*0.53*(0.5**(len(cell_name)-1))
		elif cell_name=="P3":
			v=0.45*0.46*0.47
		elif cell_name[0]=="D":
			v=0.45*0.46*0.47*0.52*(0.5**(len(cell_name)-1))
		elif cell_name=="P4":
			v=0.45*0.46*0.47*0.48
		elif cell_name in ['Z2', 'Z3']:
			v=0.45*0.46*0.47*0.48*0.5
		else:
			print('ERROR!!!!! CELL NOT FOUND IN CALCULATING HER RADIUS!!!!',cell_name)
			print('Use an average value.')
			v=v=0.55*(0.5**(9-2))		#ABarppppa

		radius = pow(self.embryo.volume * v / (4 / 3.0 * np.pi), 1/3.0)
		radius = radius

		return radius

	def get_cell_daughter(self, cell_name, cell_dict):
		daughter = []
		if cell_name == 'P0':
			daughter = ['AB', 'P1']
		elif cell_name == 'P1':
			daughter = ['EMS', 'P2']
		elif cell_name == 'P2':
			daughter = ['C', 'P3']
		elif cell_name == 'P3':
			daughter = ['D', 'P4']
		elif cell_name == 'P4':
			daughter = ['Z2', 'Z3']
		elif cell_name == 'EMS':
			daughter = ['MS', 'E']
		## standard name ###
		else:
			for cell in cell_dict.keys():
				if cell.startswith(cell_name) and len(cell) == len(cell_name) + 1:
					daughter.append(cell)
			daughter = sorted(daughter)
		if daughter == []:
			daughter = ['', '']
		return daughter

	def init_env(self):
		with open(self.file_path % self.start_point) as file:
			for line in file:
				line = line[:len(line)-1]
				vec = line.split(', ')
				id = int(vec[0])
				location = np.array((float(vec[5]), float(vec[6]), float(vec[7])))
				########### add noise to initial location##################
				location_noise = np.random.normal(0, 0.1, 2)
				location[0:2] = location[0:2] + location_noise
				########### add noise to initial location##################
				diameter = float(vec[8])
				cell_name = vec[9]
				if cell_name[0:3] == 'Nuc':
					continue
				if cell_name != '':
					self.current_cell_list.append(cell_name)
					a = CellAgent(id, self, cell_name, location, diameter)
					self.schedule.add(a)

	def set_cell_next_location(self):
		for cell in self.schedule.agents:
			if cell.cell_name in self.next_stage_destination_list:
				cell.next_location = (self.next_stage_destination_list[cell.cell_name][0:3] - cell.location) \
								/ (self.tick_resolution - self.ticks % self.tick_resolution) + cell.location
				cell.diameter = self.next_stage_destination_list[cell.cell_name][3]
			else:
				### new cell born ###
				mother = cell.cell_name
				daughter = self.get_cell_daughter(cell.cell_name, self.next_stage_destination_list)
				if daughter[0] == '':
					print('ERROR!!!!! NO DAUGHTER FOUND!!!!!')
				cell.cell_name = daughter[0]
				cell.diameter = self.next_stage_destination_list[daughter[0]][3]
				cell.next_location = (self.next_stage_destination_list[daughter[0]][0:3] - cell.location) \
								/ (self.tick_resolution - self.ticks % self.tick_resolution) + cell.location
				new_id = len(self.schedule.agents) + 1
				new_diameter = self.next_stage_destination_list[daughter[1]][3]
				a = CellAgent(new_id, self, daughter[1], cell.location, new_diameter)
				self.schedule.add(a)
				a.next_location = (self.next_stage_destination_list[daughter[1]][0:3] - a.location) \
								/ (self.tick_resolution - self.ticks % self.tick_resolution) + a.location

				self.dividing_cell_overall.append(mother)


	def update_stage_destination(self):
		current_stage_destination_point = self.start_point + 1 + int(self.ticks / self.tick_resolution)
		if self.stage_destination_point == current_stage_destination_point:
			return
		else:
			self.stage_destination_point = current_stage_destination_point
			self.next_stage_destination_list.clear()
			with open(self.file_path % self.stage_destination_point) as file:
				for line in file:
					line = line[:len(line)-1]
					vec = line.split(', ')
					id = int(vec[0])
					loc_and_dia = np.array((float(vec[5]), float(vec[6]), float(vec[7]), float(vec[8])))
					cell_name = vec[9]
					if cell_name != '':
						self.next_stage_destination_list[cell_name] = loc_and_dia

	def render(self):
		if self.ticks % FRESH_PERIOD == 0:
			self.draw(self.plane_draw)

	def reset(self):
		self.ticks = 0

		self.start_point = START_POINT
		self.end_point = END_POINT

		self.end_tick = (self.end_point - self.start_point) * self.tick_resolution
		self.stage_destination_point = self.start_point
		self.current_cell_list = []
		self.dividing_cell_overall = []
		self.next_stage_destination_list = {}

		del self.schedule.agents[:]
		self.init_env()
		self.update_stage_destination()
		s = self.get_state()

		return s

	def get_state(self):
		s = []
		low_plane = PLANE_DRAW-INPUT_PLANE_RANGE
		if low_plane <=1:
			low_plane = 1
		high_plane = PLANE_DRAW+INPUT_PLANE_RANGE+1
		for p in range(low_plane, high_plane):
			image = Image.new('RGB', (self.embryo.width-self.embryo.wid_offset, self.embryo.height-self.embryo.hei_offset))
			draw = ImageDraw.Draw(image)
			for cell in self.schedule.agents:
				if cell.cell_name == self.ai_cell:
					fill_color = 'red'
				else:
					fill_color = 'green'
				cell_loc = np.array((cell.location[0], cell.location[1], \
								cell.location[2] * self.plane_resolution))
				radius = self.get_radius(cell.cell_name)
				z_diff = cell_loc[2] - p * self.plane_resolution
				if abs(z_diff) < radius:
					radius *= 0.5
					z_diff *= 0.5
					radius_projection = (radius**2 - z_diff**2) ** 0.5
					draw.ellipse((cell_loc[0]-radius_projection-self.embryo.wid_offset,
									cell_loc[1]-radius_projection-self.embryo.hei_offset,
									cell_loc[0]+radius_projection-self.embryo.wid_offset,
									cell_loc[1]+radius_projection-self.embryo.hei_offset),
						fill = fill_color, outline ='black')

			image = image.resize((128,128))
			image_np = np.array(image).astype(np.float32) / 255			#widthxheightx3
			image_np = np.rollaxis(image_np, 2)							#3x2widthxheight
			if len(s) == 0:
				s = image_np
			else:
				s = np.concatenate((s, image_np), axis=0)

		return s

	def step(self):
		r = 0
		done = False
		sg_done = False
		if self.ticks > 0 and self.ticks % self.tick_resolution == 0:
			self.update_stage_destination()

		self.set_cell_next_location()
		self.schedule.step()
		self.ticks += 1

		s_ = self.get_state()
		ai_location = np.zeros(3)

		for cell in self.schedule.agents:
			if cell.cell_name == self.ai_cell:
				ai_location = np.array((cell.location[0], cell.location[1], \
								cell.location[2] * self.plane_resolution))

		if self.ticks == self.end_tick:
			done = True

		return s_, r, sg_done, done



class CellAgent(Agent):
	def __init__(self, unique_id, model, name, location, diameter):
		super().__init__(unique_id, model)
		self.cell_name = name
		self.location = location
		self.diameter = diameter
		self.min_dist_list = []
		self.next_location = None


	def move(self):
		speed = np.linalg.norm(self.location[0:2] - self.next_location[0:2])
		location_noise = np.random.normal(0, speed * 0.1, 2)

		self.location = self.next_location
		self.location[0:2] = self.location[0:2] + location_noise
		self.next_location = None

	def step(self):
		self.move()



class CNN_Net(nn.Module):
	def __init__(self, ):
		super(CNN_Net, self).__init__()
		self.conv1 = nn.Sequential(
			nn.Conv2d(N_CHANNEL * N_INPUT, 32, 5, 4, 2),
			nn.ReLU(),
			nn.MaxPool2d(2),
			)                               #output (32x16x16)
		self.conv2 = nn.Sequential(
			nn.Conv2d(32, 64, 3, 1, 1),
			nn.ReLU(),
			nn.MaxPool2d(2),
			)                               #output (64x8x8)
		self.out = nn.Linear(64*8*8, 8)

	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = x.view(x.size(0),-1)
		actions_value = self.out(x)
		return actions_value



if __name__ == '__main__':
	print('Using Embryo #'+str(args.emb))
	N_CHANNEL = 15
	INPUT_SIZE = 128
	N_INPUT = 1
	if torch.cuda.is_available():
		use_cuda = True
	else:
		use_cuda = False
	FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
	LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
	if use_cuda:
		eval_net = CNN_Net().cuda()
	else:
		eval_net = CNN_Net()
	eval_net.load_state_dict(torch.load('./trained_models/hdrl_llmodel.pkl', map_location=lambda storage, loc: storage))

	env = SeqRosModel()
	input_buffer = []
	movement_types = []
	s = env.reset()
	for ni in range(N_INPUT):
		input_buffer.append(s)
	s_all = []
	for ipt in input_buffer:
		if s_all == []:
			s_all = ipt
		else:
			s_all = np.concatenate((s_all, ipt), axis=0)  #s_all shape:(?*128*128)

	ep_r = 0
	counter = 0

	while True:
		env.render()
		x = np.reshape(s_all, (-1,N_CHANNEL*N_INPUT,INPUT_SIZE,INPUT_SIZE))
		if use_cuda:
			x = Variable(torch.FloatTensor(x), 0).cuda()
		else:
			x = Variable(torch.FloatTensor(x), 0)
		for layer, (name, module) in enumerate(eval_net._modules.items()):
			if isinstance(module, torch.nn.Linear):
				x = x.view(x.size(0),-1)
				x = x.data.cpu().numpy()[0]
				break
			x = module(x)

		feature = Variable(torch.Tensor(x).type(FloatTensor))
		pred = env.speed_model_plus(feature)
		pred = pred.data.cpu().numpy()
		movement_type = np.argmax(pred)
		movement_types.append(movement_type)
						
		s_, r, sg_done, done = env.step()
		del input_buffer[0]
		input_buffer.append(s_)
		s_all_ = []
		for input in input_buffer:
			if s_all_ == []:
				s_all_ = input
			else:
				s_all_ = np.concatenate((s_all_, input), axis=0)

		counter += 1
		ep_r += r

		if done:
			print('Done in', counter, 'steps.')
			print('Movement type:', movement_types)
			break
		s_all = s_all_

