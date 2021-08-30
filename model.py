from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from sklearn import svm
import _pickle as pickle

import random
import sys,time,os
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

RUN_LEARNING = True

TICK_RESOLUTION = 10
AI_CELL = 'Cpaaa'

PLANE_THRESHOLD = 3			#draw the cells in the [-2,+2] planes (5 total)
PLANE_DRAW = 8

INPUT_CENTER_PLANE = 7
INPUT_PLANE_RANGE = 2

CANVAS_DISPLAY_SCALE_FACTOR = 2		### zoom in/out the size of canvas
FRESH_TIME = 0.02
FRESH_PERIOD = 1


PLANE7_LONG_AXIS_1 = [170, 205]
PLANE7_LONG_AXIS_2 = [283, 333]
PLANE7_SHORT_AXIS_1 = [190, 303]
PLANE7_SHORT_AXIS_2 = [263, 237]

PLANE7_CENTER = [225, 275]
PLANE7_VEC0 = [1, 1]
PLANE7_VEC1 = [1, -1]
PLANE7_E0 = np.linalg.norm(np.array(PLANE7_LONG_AXIS_2) - np.array(PLANE7_CENTER))
PLANE7_E1 = np.linalg.norm(np.array(PLANE7_SHORT_AXIS_2) - np.array(PLANE7_CENTER))

EMBRYO_TOTAL_PLANE = 30
PLANE_RESOLUTION = 4

RADIUS_SCALE_FACTOR = 1.0

AI_CELL_SPEED_PER_MIN = 2

NEIGHBOR_MODEL_PATH = './trained_models/neighbor_model.pkl'

NEIGHBOR_CANDIDATE_1 = [['ABarppppa', 'ABarppapp'], ['ABarpppap', 'ABarppapp'],['ABarppppa', 'ABarppapp', 'ABarpppap']]

NEIGHBOR_CANDIDATE_2 = [['ABarpppap', 'ABarppapa'], ['ABarpppap', 'ABarppaap'], ['ABarpppaa', 'ABarppapa'], ['ABarpppaa', 'ABarppaap'], 
						['ABarpppap', 'ABarppapa', 'ABarpppaa'], ['ABarpppap', 'ABarppapa', 'ABarppaap'], 
						['ABarpppaa', 'ABarppaap', 'ABarpppap'], ['ABarpppaa', 'ABarppaap', 'ABarppapa'],
						['ABarpppap', 'ABarppapa', 'ABarpppaa', 'ABarppaap']]

NEIGHBOR_CANDIDATE_3 = [['ABarpppaa', 'ABarppaap']]

NEIGHBOR_FINAL = 'ABarpaapp'

STATE_CELL_LIST = ['ABarpppap', 'ABarppppa', 'ABarppppp', 'Caaaa', 'ABprapapp', 'Epra', 'ABprapaaa', \
                                'ABprapaap', 'Cpaap', 'ABprapapa', 'ABarppapp', 'Caaap', 'Eprp', 'ABarpppaa', 'Eplp', \
                                'ABarppapa', 'Epla', 'ABarppaap']

if torch.cuda.is_available():
    use_cuda = True
else:
    use_cuda = False

class AlexNet(nn.Module):
	def __init__(self, num_classes=2, num_channel=3, img_resolution=128):
		super(AlexNet, self).__init__()
		self.features = nn.Sequential(
			nn.Conv2d(num_channel, 64, kernel_size=11, stride=4, padding=5),	#4096->1024
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),								#1024->512
			nn.Conv2d(64, 192, kernel_size=5, padding=2),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),								#512->256
			nn.Conv2d(192, 384, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(384, 256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),								#256->128
		)
		self.classifier = nn.Sequential(
			nn.Linear(256 * int(img_resolution/32) * int(img_resolution/32), 1000),
			nn.ReLU(inplace=True),
			nn.Linear(1000, num_classes),
		)

	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x

class SeqRosModel(Model):
	def __init__(self):
		if use_cuda:
			self.motion_model = AlexNet().cuda()
			self.motion_model.load_state_dict(torch.load('./trained_models/motion_model.pkl'))
		else:
			self.motion_model = AlexNet()
			self.motion_model.load_state_dict(torch.load('./trained_models/motion_model.pkl', map_location='cpu'))
		self.file_path = './data/cpaaa_0/t%03d-nuclei'
		self.start_point = 168
		self.end_point = 197
		print('Parsing the Embryo...')
		self.embryo = Embryo('./data/cpaaa_0/')
		self.embryo.read_data()
		self.embryo.get_embryo_visual_params()
		self.embryo.store_ai_cell_observed_locations(start=self.start_point)
		################# time consuming #########################
		# self.embryo.parse_all()
		self.embryo.x1 = np.array([133.0, 174.0, 78.0])
		self.embryo.x2 = np.array([330.0, 365.0, 78.0])
		self.embryo.x_length = 274.390233062
		self.embryo.y1 = np.array([255.0, 222.0, 84.0])
		self.embryo.y2 = np.array([156.0, 328.0, 84.0])
		self.embryo.y_length = 145.016094618
		self.embryo.volume = 2500578.08321

		self.alpha = np.arccos( (330-133) / np.linalg.norm(self.embryo.x1-self.embryo.x2) ) + np.pi/16
		self.center = np.array([(330-133)/2, (365-174)/2])
		################# time consuming #########################

		self.ai_cell = AI_CELL

		n_neighbor_choice = np.random.randint(len(NEIGHBOR_CANDIDATE_1))
		self.neighbor_list_goal = NEIGHBOR_CANDIDATE_1[n_neighbor_choice]
		self.current_subgoal_index = 0

		self.neighbor_goal_counter = 0
		self.neighbor_goal_achieved_num = 15

		self.neighbor_model = pickle.load(open(NEIGHBOR_MODEL_PATH, 'rb'))

		self.ticks = 0
		self.tick_resolution = TICK_RESOLUTION
		self.end_tick = (self.end_point - self.start_point) * self.tick_resolution
		self.stage_destination_point = self.start_point

		self.embryo_total_plane = EMBRYO_TOTAL_PLANE
		self.plane_resolution = PLANE_RESOLUTION

		self.radius_scale_factor = RADIUS_SCALE_FACTOR

		self.current_cell_list = []
		self.dividing_cell_overall = []
		self.next_stage_destination_list = {}
		self.state_cell_list = STATE_CELL_LIST
		self.state_value_dict = {}
		self.ai_locations = []
		self.target_locations = []
		self.schedule = RandomActivation(self)

		self.init_env()
		self.update_stage_destination()


		self.plane = DrawPlane(width=self.embryo.width,
							height=self.embryo.height,
							w_offset = self.embryo.wid_offset,
							h_offset = self.embryo.hei_offset+70,
							scale_factor = CANVAS_DISPLAY_SCALE_FACTOR)
		# self.plane = DrawPlane(width=self.embryo.width,
		# 					height=self.embryo.height,
		# 					w_offset = 100,
		# 					h_offset = 100,
		# 					scale_factor = CANVAS_DISPLAY_SCALE_FACTOR)
		self.canvas = self.plane.canvas
		self.plane_draw = PLANE_DRAW
		self.draw(self.plane_draw)


		self.ai_cell_speed = AI_CELL_SPEED_PER_MIN / float(TICK_RESOLUTION)
		self.movement_types = []

	def dist_point_ellipse(self, old_point, origin=PLANE7_CENTER, direc_vec0=PLANE7_VEC0, direc_vec1=PLANE7_VEC1):
		origin = np.array(origin)
		old_point = np.array(old_point)
		direc_vec0 = np.array(direc_vec0)
		direc_vec1 = np.array(direc_vec1)

		vec = old_point - origin
		vec_norm = np.linalg.norm(vec)
		direc_vec0_norm = np.linalg.norm(direc_vec0)
		direc_vec1_norm = np.linalg.norm(direc_vec1)
		cos_vec0 = vec.dot(direc_vec0) / vec_norm / direc_vec0_norm
		cos_vec1 = vec.dot(direc_vec1) / vec_norm / direc_vec1_norm
		x0 = vec_norm * cos_vec0
		x1 = vec_norm * cos_vec1
		new_point = np.array((x0,x1))

		new_point = np.abs(new_point)

		distance = -1.0
		if new_point[1] > 0:
			if new_point[0] > 0:
				###compute the unique root tbar of F(t) on (-e1*e1, +inf)
				t1 = - PLANE7_E1 * PLANE7_E1
				t2 = 1000000
				while True:
					ft_mid = (PLANE7_E0 * new_point[0] / (0.5 * (t1 + t2) + PLANE7_E0 ** 2)) ** 2 \
							+(PLANE7_E1 * new_point[1] / (0.5 * (t1 + t2) + PLANE7_E1 ** 2)) ** 2 - 1
					if ft_mid > 0:
						t1 = 0.5 * (t1 + t2)
					elif ft_mid < 0:
						t2 = 0.5 * (t1 + t2)
					else:
						break
					if np.abs(t1-t2)<0.001:
						break
				tbar = 0.5 * (t1 + t2)
				x0 = PLANE7_E0 * PLANE7_E0 * new_point[0] / (tbar + PLANE7_E0 * PLANE7_E0)
				x1 = PLANE7_E1 * PLANE7_E1 * new_point[1] / (tbar + PLANE7_E1 * PLANE7_E1)
				distance = np.sqrt((x0 - new_point[0]) ** 2 + (x1 - new_point[1]) ** 2)
			else:
				x0 = 0
				x1 = PLANE7_E1
				distance = np.abs(new_point[1] - PLANE7_E1)
		else:
			if new_point[0] < (PLANE7_E0 ** 2 - PLANE7_E1 ** 2) / PLANE7_E0:
				x0 = PLANE7_E0 * PLANE7_E0 * new_point[0] / (PLANE7_E0 ** 2 - PLANE7_E1 ** 2)
				x1 = PLANE7_E1 * np.sqrt(1 - (x0 / PLANE7_E0) ** 2)
				distance = np.sqrt((x0 - new_point[0]) ** 2 + x1 ** 2)
			else:
				x0 = PLANE7_E0
				x1 = 0
				distance = np.abs(new_point[0] - PLANE7_E0)

		return distance

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
				elif cell.cell_name == 'ABarpaapp':
					type = 'DEST'
				elif cell.cell_name in self.neighbor_list_goal:
					type = 'GOAL'
					level = self.neighbor_goal_counter / self.neighbor_goal_achieved_num
				else:
					type = 'NUMB'
				if round(cell.location[2]) == n:
					d_dist = np.linalg.norm(cell.location[0:2] - self.center)
					theta = self.alpha - np.arcsin( (cell.location[1]-self.center[1]) / d_dist)
					d_loc = np.array([d_dist*np.cos(theta)+20, d_dist*np.sin(theta)+300])
					
					self.plane.draw_cell(center=d_loc,
										radius=cell.diameter/2.0*np.cos(angle),
										type=type,
										level=level)

		self.canvas.pack()
		self.canvas.update()
		time.sleep(FRESH_TIME)

	def _get_ratio(self, cn):
		r=0.00000001
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
		elif cn in ['Z2', 'Z3']:
			r=0.45*0.46*0.47*0.48*0.5
		else:
			print('ERROR!!!!! CELL NOT FOUND!!!!', cn)

		return r

	def radis_ratio(self, cn):
		r = self._get_ratio(cn)
		return r**(1.0/3)

	def get_radius(self, cell_name):
		v = self._get_ratio(cell_name)
		radius = pow(self.embryo.volume * v / (4 / 3.0 * np.pi), 1/3.0)
		radius = radius * self.radius_scale_factor

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
				if cell_name != '':
					self.current_cell_list.append(cell_name)
					if cell_name == self.ai_cell:
						noise = np.random.normal(0, 0.5, 2)
						location[0:2] = location[0:2] + location_noise

					a = CellAgent(id, self, cell_name, location, diameter)
					self.schedule.add(a)

	def set_cell_next_location(self,ai_action):
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

			if RUN_LEARNING and cell.cell_name == self.ai_cell:
				offset = self.get_ai_next_location_offset(ai_action)
				cell.next_location[0:2] = cell.location[0:2] + offset[0:2]

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

	def reset(self, subgoals):
		self.ticks = 0
		self.neighbor_goal_counter = 0
		self.subgoals = subgoals
		self.current_subgoal_index = 0
		self.neighbor_list_goal = self.subgoals[self.current_subgoal_index]
		self.start_point = 168
		self.end_point = 197
		self.movement_types = []

		self.end_tick = (self.end_point - self.start_point) * self.tick_resolution
		self.stage_destination_point = self.start_point
		self.current_cell_list = []
		self.dividing_cell_overall = []
		self.next_stage_destination_list = {}
		self.state_value_dict = {}

		del self.schedule.agents[:]
		self.ai_locations = []
		self.target_locations = []
		self.init_env()
		self.update_stage_destination()
		s = self.get_state()

		return s

	def get_state(self):
		s = []
		low_plane = INPUT_CENTER_PLANE-INPUT_PLANE_RANGE
		if low_plane <=1:
			low_plane = 1
		high_plane = INPUT_CENTER_PLANE+INPUT_PLANE_RANGE+1
		for p in range(low_plane, high_plane):
			image = Image.new('RGB', (self.embryo.width-self.embryo.wid_offset, self.embryo.height-self.embryo.hei_offset))
			draw = ImageDraw.Draw(image)
			for cell in self.schedule.agents:
				if cell.cell_name == self.ai_cell:
					fill_color = 'red'
				elif cell.cell_name in self.neighbor_list_goal:
					yellow_level = int(self.neighbor_goal_counter / self.neighbor_goal_achieved_num * 50) + 200
					fill_color = '#%02x%02x%02x' % (yellow_level, yellow_level, 0)
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
			if s == []:
				s = image_np
			else:
				s = np.concatenate((s, image_np), axis=0)
		return s

	def get_img_motion_model(self):
		img_resolution = 128
		for cell in self.schedule.agents:
			if cell.cell_name == self.ai_cell:
				plane = int(cell.location[2])
				ai_cell_loc = np.array((cell.location[0], cell.location[1], cell.location[2] * self.plane_resolution))

		image = Image.new('RGB', (img_resolution, img_resolution))
		draw = ImageDraw.Draw(image)

		for cell in self.schedule.agents:
			if cell.cell_name == self.ai_cell:
				fill_color = 'red'
			else:
				fill_color = 'green'
			cell_loc = np.array((cell.location[0], cell.location[1], cell.location[2] * self.plane_resolution))
			if abs(cell_loc[0] - ai_cell_loc[0]) > 0.5 * img_resolution or abs(cell_loc[1] - ai_cell_loc[1]) > 0.5 * img_resolution:
				continue
			radius = self.get_radius(cell.cell_name)
			z_diff = cell_loc[2] - plane * self.plane_resolution
			if abs(z_diff) < radius:
				radius *= 0.5
				z_diff *= 0.5
				radius_projection = (radius**2 - z_diff**2) ** 0.5				

				draw.ellipse((cell_loc[0]-radius_projection-ai_cell_loc[0]+0.5*img_resolution,
								cell_loc[1]-radius_projection-ai_cell_loc[1]+0.5*img_resolution,
								cell_loc[0]+radius_projection-ai_cell_loc[0]+0.5*img_resolution,
								cell_loc[1]+radius_projection-ai_cell_loc[1]+0.5*img_resolution),
							fill = fill_color, outline ='black')

		image = np.rollaxis(np.array(image), 2)
		image = image[np.newaxis,:]
		if use_cuda:
			FloatTensor = torch.cuda.FloatTensor 
			LongTensor = torch.cuda.LongTensor
		else:
			FloatTensor = torch.FloatTensor 
			LongTensor = torch.LongTensor
		image = Variable(torch.from_numpy(image).type(FloatTensor))

		pred = self.motion_model(image)
		pred = pred.data.cpu().numpy()
		result = np.argmax(pred, axis=1)[0]

		return result

	def get_reward(self):
		r = 0
		done = False
		sg_done = False
		ai_location = np.copy(self.state_value_dict[self.ai_cell])
		ai_location[2] = ai_location[2] * self.plane_resolution
		ai2bdr_dist = self.dist_point_ellipse(ai_location[0:2])
		ai_radius = self.get_radius(self.ai_cell)

		movement_type = self.get_img_motion_model()
		if movement_type == 1:
			self.movement_types.append(1)
			r += 1
		else:
			self.movement_types.append(0)

		#########boundary control rule: (1) dist>0.7*radius, ok  (2)0.4*r<dist<0.7*r, bad    (3)dist<0.4*r, dead
		if ai2bdr_dist > 0.7 * ai_radius:
			r += 0
		elif ai2bdr_dist > 0.4 * ai_radius and ai2bdr_dist <= 0.7 * ai_radius:
			r += (0.7 - float(ai2bdr_dist) / ai_radius) / (0.4 - 0.7)		## 0.5->-1, 0.8->0
		elif ai2bdr_dist < 0.4 * ai_radius:
			r = -1000
			done = True
			print('hit boundary')
			return r, sg_done, done

		######### pressure with other cells control rule ################
		######### (1) dist>0.8*r, ok  (2)0.3*r<dist<0.8*r, bad    (3)dist<0.3*r, dead
		for item in self.state_value_dict.keys():
			# print(r)
			if item != self.ai_cell:
				cell_location = np.copy(self.state_value_dict[item])
				cell_location[2] = cell_location[2] * self.plane_resolution
				dist = np.linalg.norm(cell_location - ai_location)
				cell_radius = self.get_radius(item)
				sum_radius = cell_radius + ai_radius
				dead_factor = 0.4
				ok_factor = 0.7
				if dist > ok_factor * sum_radius:
					r += 0
				elif dist > dead_factor * sum_radius and dist <= ok_factor * sum_radius:
					r += (ok_factor - float(dist) / sum_radius) / (dead_factor - ok_factor)		## 0.4->-1, 0.6->0

				elif dist < dead_factor * sum_radius:
					print('hit other cell:', item)
					r = -1000
					done = True
					return r, sg_done, done

		if self.neighbor_goal_counter == self.neighbor_goal_achieved_num:
			if NEIGHBOR_FINAL in self.neighbor_list_goal:
				r = 100
				done = True
				print('Final goal is achieved. Done!')
				return r, sg_done, done
			else:
				sg_done = True
				r = 10
				print('Subgoal Done:' + str(self.neighbor_list_goal))
				# set next sub-goal
				return r, sg_done, done

		if r == 0:
			r = 1

		return r, sg_done, done



	def get_ai_next_location_offset(self,a):
		offset_45 = self.ai_cell_speed * pow(2,0.5) / 2.0
		if a == 0:
			offset = np.array((0, - self.ai_cell_speed, 0))
		elif a == 1:
			offset = np.array((0, self.ai_cell_speed, 0))
		elif a == 2:
			offset = np.array((- self.ai_cell_speed, 0, 0))
		elif a == 3:
			offset = np.array((self.ai_cell_speed, 0, 0))
		elif a == 4:
			offset = np.array((- offset_45, -offset_45, 0))
		elif a == 5:
			offset = np.array((offset_45, -offset_45, 0))
		elif a == 6:
			offset = np.array((- offset_45, offset_45, 0))
		elif a == 7:
			offset = np.array((offset_45, offset_45, 0))
		elif a == 8:
			offset = np.array((0, 0, 0))
		return offset

	def step(self, a):
		#self.get_img_speed_model()
		done = False
		sg_done = False
		if self.ticks > 0 and self.ticks % self.tick_resolution == 0:
			self.update_stage_destination()

		self.set_cell_next_location(ai_action=a)
		self.schedule.step()
		self.ticks += 1

		s_ = self.get_state()
		ai_location = np.zeros(3)
		target_loc = np.zeros(3)
		neighbor_goal_locations = {}
		for cell in self.schedule.agents:
			if cell.cell_name == self.ai_cell:
				ai_location = np.array((cell.location[0], cell.location[1], \
								cell.location[2] * self.plane_resolution))
				self.ai_locations.append(ai_location.round(2))
			elif cell.cell_name in self.neighbor_list_goal:
				loc = np.array((cell.location[0], cell.location[1], \
								cell.location[2] * self.plane_resolution))
				neighbor_goal_locations[cell.cell_name] = loc
		for cell in self.schedule.agents:
			if cell.cell_name == 'ABarpaapp':
				target_loc = np.array((cell.location[0], cell.location[1], \
								cell.location[2] * self.plane_resolution))
				self.target_locations.append(target_loc.round(2))
		is_goal_achieved = True
		for cell in neighbor_goal_locations.keys():
			dist = np.linalg.norm(ai_location - neighbor_goal_locations[cell])
			r_ai_cell = self.radis_ratio(self.ai_cell)
			r_cell = self.radis_ratio(cell)
			is_neighbor = self.neighbor_model.predict([[dist, r_ai_cell, r_cell, len(self.schedule.agents)]])
			if is_neighbor == 0:
				is_goal_achieved = False
				break
		if is_goal_achieved:
			if self.neighbor_goal_counter < self.neighbor_goal_achieved_num:
				self.neighbor_goal_counter += 1

		r, sg_done, done = self.get_reward()
		if self.ticks == self.end_tick:
			done = True
			r = 0
			print('Time is up! Goal NOT achieved!')
		if self.current_subgoal_index == 0 and self.ticks > 70:	#cannot achieve subgoal within certain time
			done = True 
			r = 0
			print('Time is up! sub-goal #1 NOT achieved!')
		if sg_done:
			self.current_subgoal_index += 1
			if self.current_subgoal_index < len(self.subgoals):
				self.neighbor_list_goal = self.subgoals[self.current_subgoal_index]
			else:
				self.neighbor_list_goal = [NEIGHBOR_FINAL]
			self.neighbor_goal_counter = 0

		return s_, r, sg_done, done


class CellAgent(Agent):
	def __init__(self, unique_id, model, name, location, diameter):
		super().__init__(unique_id, model)
		self.cell_name = name
		self.location = location
		self.diameter = diameter
		self.min_dist_list = []
		self.next_location = None

		self.set_state()

	def set_state(self):
		if self.cell_name == self.model.ai_cell:
			self.model.state_value_dict[self.cell_name] = self.location
		elif self.cell_name in self.model.state_cell_list:
			self.model.state_value_dict[self.cell_name] = self.location



	def move(self):
		speed = np.linalg.norm(self.location[0:2] - self.next_location[0:2])
		location_noise = np.random.normal(0, speed * 0.1, 2)

		self.location = self.next_location
		self.location[0:2] = self.location[0:2] + location_noise
		self.next_location = None

	def step(self):
		self.move()
		self.set_state()


