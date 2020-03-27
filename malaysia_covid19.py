import os
import conda
conda_file_dir = conda.__file__
conda_dir = conda_file_dir.split('lib')[0]
proj_lib = os.path.join(os.path.join(conda_dir, 'share'), 'proj')
os.environ["PROJ_LIB"] = proj_lib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
import matplotlib

class time_series:
	def __init__ (self):
		self.data = [0]

	def import_file(self):
		url1 = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
		url2 = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
		url3 = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'

		data = pd.read_csv(url1)
		data2 = data[data['Country/Region'] == 'Malaysia']

		data_1 = pd.read_csv(url2)
		data_1_1 = data_1[data_1['Country/Region'] == 'Malaysia']

		data_2 = pd.read_csv(url3)
		data_2_1 = data_2[data_2['Country/Region'] == 'Malaysia']

		datatemp = pd.merge(data2, data_1_1, how = 'outer')
		datatemp = pd.merge(datatemp, data_2_1, how = 'outer')

		data3 = {'Dates' : np.array(datatemp.iloc[0, 4:].index), 'Cases' : np.array(datatemp.iloc[0, 4:].values), 'Deaths' : np.array(datatemp.iloc[1, 4:].values),
				 'Recovered' : np.array(datatemp.iloc[2, 4:].values)}

		df = pd.DataFrame(data3, columns = ['Dates','Cases','Deaths','Recovered'])
		df['Dates'] = pd.to_datetime(df['Dates'])
		df['Dates'] = df['Dates'].dt.strftime('%d/%m/%Y')
		df.set_index('Dates', inplace=True)
		self.data = df

	def plot_graph(self):
		new_x = np.arange(0, len(self.data.index), step=1)
		new_x = np.array(new_x, dtype=float)
		new_y1 = np.array(self.data['Cases'].values, dtype = float)
		new_y2 = np.array(self.data['Deaths'].values, dtype = float)
		new_y3 = np.array(self.data['Recovered'].values, dtype = float)

		plt.style.use('seaborn')
		fig, ax1 = plt.subplots()
		ax1.set_prop_cycle(color = ['darkred', 'mediumvioletred', 'olivedrab'])
		ax1.plot(self.data)
		ax1.set_xticks(np.arange(0, len(self.data.index), step=7))
		fig.autofmt_xdate()
		ax1.legend(['Cases','Deaths','Recovered'])
		ax1.set_xlabel('Dates')
		ax1.set_ylabel('Confirmed Cases')

		ax1.fill_between(new_x, 0 , new_y1, color = 'red', alpha = 0.5)
		ax1.fill_between(new_x, 0 , new_y3, color = 'yellowgreen', alpha = 0.7)
		ax1.fill_between(new_x, 0 , new_y2, color = 'violet', alpha = 0.8)

		plt.show()
		
	def plot_bar(self):
		plt.style.use('seaborn')
		fig, ax1 = plt.subplots()
		ax1.set_prop_cycle(color = ['darkred', 'olivedrab', 'mediumvioletred'])
		ax1.bar(self.data.index, self.data['Cases'], alpha = 0.85)
		ax1.bar(self.data.index, self.data['Recovered'], alpha = 0.85)
		ax1.bar(self.data.index, self.data['Deaths'], alpha = 0.85)
		ax1.set_xticks(np.arange(0, len(self.data.index), step=7))
		fig.autofmt_xdate()
		ax1.legend(['Cases','Recovered','Deaths'])
		ax1.set_xlabel('Dates')
		ax1.set_ylabel('Confirmed Cases')

		plt.show()

class case_distribution():
	def __init__ (self):
		self.data_1 = [0]
		self.data_2 = [0]

	def import_file(self):
		data = pd.read_csv('kescov2.csv', index_col = 'Date', parse_dates = True)
		data2 = data.cumsum()

		data1 = data2.sort_values(by = data.index[-1], axis = 1, ascending = False)
		self.data_1 = data1.ffill()

		data2 = data2.reset_index()
		data2 = data2.drop('Date', axis = 1)
		data2 = data2.T
		data2.index.name = 'state'
		self.data_2 = data2

	def worst_3(self):
		plt.style.use('seaborn')
		fig, ax = plt.subplots()
		label = [self.data_1.columns[n] for n in range(3)]
		for n in range(3):
			ax.plot(self.data_1.iloc[:, n])
		fig.autofmt_xdate()
		ax.set_xlabel('Dates')
		ax.set_ylabel('Confirmed Cases')
		ax.legend(label)

		plt.show()

	def case_map(self):
		fig, ax = plt.subplots(figsize=(200,20))
		m = Basemap(width=8E6, height=8E6,lat_0 = -5, lon_0 = 110, llcrnrlon=99, llcrnrlat=-2,
    		urcrnrlon=119.8, urcrnrlat=9, resolution = 'l')
		m.drawmapboundary(fill_color='white')
		m.fillcontinents(color='gray',lake_color='aqua')

		m.readshapefile('/home/hakimvira/Documents/python/my/MYS_adm1', 'my', drawbounds = True, linewidth=0.5)

		df_poly = pd.DataFrame({
		        'shapes': [Polygon(np.array(shape), True) for shape in m.my],
		        'area': [area['NAME_1'] for area in m.my_info] })

		df_poly = df_poly.merge(self.data_2, left_on='area', right_on = 'state', how='left')
			
		cmap = plt.get_cmap('Reds')   
		pc = PatchCollection(df_poly.shapes, zorder=2)
		norm = Normalize()
		pc.set_facecolor(cmap(norm(df_poly.iloc[:,-1].fillna(0).values)))
		ax.add_collection(pc)

		mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
		 
		mapper.set_array(df_poly.iloc[:,-1])
		plt.colorbar(mapper, shrink=0.4)

		plt.show()

j = time_series()
j.import_file()
j.plot_graph()
j.plot_bar()

k = case_distribution()
k.import_file()
k.worst_3()
k.case_map()
