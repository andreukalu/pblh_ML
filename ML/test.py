from utilities import *

df = pd.read_pickle('predictions')
print(df.shape)
x = df['og']*3500-df['min_altitude_rs']*3500
y = df['pred']*3500-df['min_altitude_rs']*3500

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
rmse = np.sqrt(np.sum((x-y)**2)/len(x))
md = np.sum(x-y)/len(x)
n = len(x)
ablhmean = np.mean(x)

plt.figure(figsize=(5,4))
plt.scatter(x,y,s=2,alpha=0.5)
plt.xlim([0,3500])
plt.ylim([0,3500])
plt.grid('on')
plt.plot([0,3500],[0,3500],linestyle='--',color='k',linewidth=1)
lrline = np.array([intercept,slope*3500+intercept])
lr = plt.plot(np.array([0,3500]),lrline,color='r')
plt.xlabel('$PBLH_{RS}$ [m]')
plt.ylabel('$PBLH_{CALIPSO}$ [m]')
plt.text(1800,500,'$R^2$=' + str(np.round(r_value**2,2)))
plt.text(1800,800,'$RMSE$=' + str(np.round(rmse,2)) + ' m')
plt.text(1800,200,'$MD$=' + str(np.round(md,2)) + ' m')
plt.text(1800,1100,'$N$=' + str(np.round(n,2)))
plt.text(1800,1400,'$\overline{PBLH}_{RS}$=' + str(np.round(ablhmean,2)) + ' m')

plt.legend(lr,['$y=' + str(round(slope,2)) + '\cdot x+' + str(round(intercept,2)) + '$'])
plt.savefig('/home/usuaris/csl/andreu.salcedo/ABLH/01Code/ML/scatter.png', dpi=300)
print('saved')


df = pd.read_pickle('predictions_train')
print(df.shape)
x = df['og']*3500#-df['min_altitude_rs']*3500
y = df['pred']*3500#-df['min_altitude_rs']*3500

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
rmse = np.sqrt(np.sum((x-y)**2)/len(x))
md = np.sum(x-y)/len(x)
n = len(x)
ablhmean = np.mean(x)

plt.figure(figsize=(5,4))
plt.scatter(x,y,s=2,alpha=0.5)
plt.xlim([0,3500])
plt.ylim([0,3500])
plt.grid('on')
plt.plot([0,3500],[0,3500],linestyle='--',color='k',linewidth=1)
lrline = np.array([intercept,slope*3500+intercept])
lr = plt.plot(np.array([0,3500]),lrline,color='r')
plt.xlabel('$PBLH_{RS}$ [m]')
plt.ylabel('$PBLH_{CALIPSO}$ [m]')
plt.text(1800,500,'$R^2$=' + str(np.round(r_value**2,2)))
plt.text(1800,800,'$RMSE$=' + str(np.round(rmse,2)) + ' m')
plt.text(1800,200,'$MD$=' + str(np.round(md,2)) + ' m')
plt.text(1800,1100,'$N$=' + str(np.round(n,2)))
plt.text(1800,1400,'$\overline{PBLH}_{RS}$=' + str(np.round(ablhmean,2)) + ' m')

plt.legend(lr,['$y=' + str(round(slope,2)) + '\cdot x+' + str(round(intercept,2)) + '$'])
plt.savefig('/home/usuaris/csl/andreu.salcedo/ABLH/01Code/ML/scatter_train.png', dpi=300)
print('saved')

# Plotting predictions vs true values
plt.figure(figsize=(10, 6))
plt.scatter(x, y, s=3, alpha=0.5)
plt.colorbar()
plt.plot([min(x), max(x)], [min(x), max(x)], 'r--', lw=2)
plt.title('Predictions vs True Values')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid()
plt.gca().set_aspect('equal', adjustable='box')

# Save the figure
plt.savefig('/home/usuaris/csl/andreu.salcedo/ABLH/01Code/ML/predictions_vs_true_values.png', dpi=300)