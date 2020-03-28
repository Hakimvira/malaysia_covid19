import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import urllib.request as request
import json
import plotly.express as px

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
		self.data_0 = [0]
		self.data_1 = [0]
		self.data_2 = [0]
		self.my = [0]

	def import_file(self):
		url1 = 'https://raw.githubusercontent.com/Hakimvira/malaysia_covid19/master/my2.geojson'
		url2 = 'https://raw.githubusercontent.com/Hakimvira/malaysia_covid19/master/kes_harian.csv'

		with request.urlopen(url1) as response:
        		source = response.read()
        		self.my = json.loads(source)	

		data = pd.read_csv(url2, index_col = 'Date', parse_dates = True)
		date = pd.read_csv(url2)
		date = np.array(date['Date'].values)

		data2 = data.cumsum()

		data1 = data2.sort_values(by = data.index[-1], axis = 1, ascending = False)
		self.data_0 = data.ffill()
		self.data_1 = data1.ffill()

		data2 = data2.reset_index()
		data2 = data2.drop('Date', axis = 1)
		data2 = data2.T
		data2.index.name = 'state'
		data2 = data2.reset_index()

		for n,d in zip(np.array(data2.columns.values[1:]),date):
	    		data2 = data2.rename(columns = { n : d })

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

	def state(self,state):
		for n in self.data_2['state'].values:
			if str(state) == n:
				data = self.data_2[self.data_2['state'] == state]
				data = data.ffill(axis = 1)
				data = data.iloc[0,1:].values
				
				plt.style.use('seaborn')
				fig, ax = plt.subplots()
				ax.plot(self.data_1.index, self.data_0[state], color = 'orange', label = 'daily')
				ax.plot(self.data_1.index, data, color = 'red', label = 'total')
				fig.autofmt_xdate()
				ax.legend()
				plt.show()
			
			else:
				continue
	
	def case_map(self):
		fig = px.choropleth(self.data_2, geojson=self.my, locations= 'state', featureidkey='properties.name', color= self.data_2.iloc[:,-1].name,
                           color_continuous_scale="Reds",
                           scope = 'asia', labels = {'self.data_2.iloc[:,-1].name' : 'Cases'})

		fig.update_geos(fitbounds="locations", visible=False)
		fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

		fig.show()

j = time_series()
j.import_file()
j.plot_graph()
j.plot_bar()

k = case_distribution()
k.import_file()
k.worst_3()
k.state('Melaka')
k.case_map()
