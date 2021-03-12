import numpy as np
import re

class Embryo(object):
	def __init__(self, 
				file_path='./nuclei/', 
				start_point=1, 
				end_point=200, 
				total_plane=30, 
				z_resolution=0.25):
		self.raw_data = []
		self.ai_cell_observed_locations = []
		self.file_path = file_path
		self.start_point = start_point
		self.end_point = end_point
		self.total_plane = total_plane
		self.z_resolution = z_resolution

		self.width = -1
		self.height = -1
		self.wid_offset = -1
		self.hei_offset = -1

		self.x1 = -1
		self.x2 = -1
		self.x_length = -1
		self.y1 = -1
		self.y2 = -1	
		self.y_length = -1	
		self.volume = -1

	def parse_all(self):
		self.read_data()
		self.get_embryo_visual_params()
		self.get_volume()

	def read_data(self):
		print('Reading data...')
		for i in range(self.start_point, self.end_point+1):
			self.raw_data.append([])
			file_name = ''
			if i in range(0,10):
				file_name = 't00' + str(i) + '-nuclei'
			elif i in range(10, 100):
				file_name = 't0' + str(i) + '-nuclei'
			elif i in range(100,1000):
				file_name = 't' + str(i) + '-nuclei'
			with open(self.file_path+file_name, 'r') as file:
				for line in file:
					line = line[:len(line)-1]
					vec = re.split(", ", line)
					self.raw_data[i-1].append(vec[5:10])
		# print(self.raw_data[15])

	def store_ai_cell_observed_locations(self, cell='Cpaaa', start=168, end=197):
		for i in range(start, end+1):
			file_name = ''
			if i in range(0,10):
				file_name = 't00' + str(i) + '-nuclei'
			elif i in range(10, 100):
				file_name = 't0' + str(i) + '-nuclei'
			elif i in range(100,1000):
				file_name = 't' + str(i) + '-nuclei'
			with open(self.file_path+file_name, 'r') as file:
				for line in file:
					line = line[:len(line)-1]
					vec = re.split(", ", line)
					if vec[9] == cell:
						location = np.array((float(vec[5]), float(vec[6]), float(vec[7])))
						self.ai_cell_observed_locations.append(location)
						


	def get_embryo_visual_params(self, r=15, tolerance=10):
		print('Getting visualaztion parameters...')
		x_min = 10000
		y_min = 10000
		x_max = 0
		y_max = 0
		for step_data in self.raw_data:
			for sample in step_data:
				if float(sample[0]) < x_min:
					x_min = float(sample[0])
				if float(sample[0]) > x_max:
					x_max = float(sample[0])
				if float(sample[1]) < y_min:
					y_min = float(sample[1])
				if float(sample[1]) > y_max:
					y_max = float(sample[1])
		
		self.width = int(x_max+r+tolerance)
		self.height = int(y_max+r+tolerance)
		self.wid_offset = int(x_min-r-tolerance)
		self.hei_offset = int(y_min-r-tolerance)
		print('Width:',self.width)
		print('Height:',self.height)
		print('Width offset:',self.wid_offset)
		print('Height offset:',self.hei_offset)
	
	def get_volume(self):
		print('Parsing axes lengths and embryo volume...')
		x_axis_length = -1
		x_c1 = -1
		x_c2 = -1
		for step_data in self.raw_data:
			for i in range(len(step_data)):
				c1 = np.array([float(step_data[i][0]), float(step_data[i][1]), float(step_data[i][2])/self.z_resolution])
				for j in range(i+1,len(step_data)):
					c2 = np.array([float(step_data[j][0]), float(step_data[j][1]), float(step_data[j][2])/self.z_resolution])
					dist = np.linalg.norm(c1-c2)
					if dist > x_axis_length:
						x_axis_length = dist
						x_c1 = c1
						x_c2 = c2
		z_avg = 0.5 * (x_c1[2]+x_c2[2])
		x_c1[2] = z_avg
		x_c2[2] = z_avg
		x_axis_length = np.linalg.norm(x_c1-x_c2)
		print('Long-axis length:',x_axis_length)
		print('Long-axis end points:',x_c1.tolist(),x_c2.tolist())

		self.x1 = x_c1
		self.x2 = x_c2
		self.x_length = x_axis_length

		x_direction = ((x_c1-x_c2) / x_axis_length)
		y_direction = [-x_direction[1],x_direction[0],x_direction[2]]

		y_axis_length = -1
		y_c1 = -1
		y_c2 = -1

		for step_data in self.raw_data:
			for i in range(len(step_data)):
				c1 = np.array([float(step_data[i][0]), float(step_data[i][1]), float(step_data[i][2])/self.z_resolution])
				for j in range(i+1,len(step_data)):
					c2 = np.array([float(step_data[j][0]), float(step_data[j][1]), float(step_data[j][2])/self.z_resolution])
					dist_len = np.linalg.norm(c1-c2)
					dist_project = np.dot(y_direction, (c1-c2)/dist_len) * dist_len
					# if dist_len<np.abs(dist_project):
					# 	print(dist_len,dist_project)
					if dist_project > y_axis_length:
						y_axis_length = dist_project
						y_c1 = c1
						y_c2 = c2
		print(y_c1,y_c2)
		print(y_axis_length)
		z_avg = 0.5 * (y_c1[2]+y_c2[2])
		y_c1[2] = z_avg
		y_c2[2] = z_avg
		y_axis_length = np.linalg.norm(y_c1-y_c2)
		print('Short-axis length:',y_axis_length)
		print('Short-axis end points:',y_c1.tolist(),y_c2.tolist())

		self.y1 = y_c1
		self.y2 = y_c2
		self.y_length = y_axis_length

		self.volume = 4 / 3.0 * np.pi \
						* (0.5 * self.x_length) \
						* (0.5 * self.y_length) \
						* (0.5 * self.total_plane / self.z_resolution)
		print('Embryo volume:',self.volume)

	# def get_plane_info(self, n_plane):
	# 	center = -1
	# 	x1 = -1
	# 	x2 = -1
	# 	y1 = -1
	# 	y2 = -1
	# 	x_length = -1
	# 	y_length = -1
	# 	for step_data in self.raw_data:
	# 		for i in range(len(step_data)):

	# 			for j in range(i+1,len(step_data)):




if __name__ == '__main__':
	em = Embryo('./nuclei/')
	em.read_data()
	em.get_volume()
	# width, height, wid_offset, hei_offset = em.get_embryo_visual_params()
	# print(width, height, wid_offset, hei_offset)

