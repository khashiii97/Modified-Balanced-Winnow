from src import MBW
import  data
import  src.config as config
import numpy as np

alphas = [1.2, 1.5, 1.8]
betas = [0.1, 0.2, 0.5, 0.8, 0.9]
ms = [1]
controller = data.DataController(batch_size=1, data_list=config.t_labels)
# generating validation and test data
X_val = []
y_val = []
while 1:
    sample = controller.generate(mode='validation')
    if sample is False:
        break
    X_val += list(sample['x'])
    y_val += list(sample['x'])
X_val = np.array(X_val)
y_val = np.array(y_val)

controller.reset()
X_test = []
y_test = []
while 1:
    sample = controller.generate(mode='test')
    if sample is False:
        break
    X_test += list(sample['x'])
    y_test += list(sample['x'])
X_test = np.array(X_test)
y_test = np.array(y_test)

best_model_based_on_f1 = ''
best_f1 = 0
best_alpha_f1 = 0
best_beta_f1 = 0
best_m = 0
for m in ms:
    for alpha in alphas:
        for beta in betas:
            model = MBW.MBWinnow(alpha=alpha, beta=beta, M=m)
            controller = data.DataController(batch_size=1, data_list=config.t_labels)
            while 1:
                sample = controller.generate('train')
                if sample is False:
                    break
                model.train(X_train=[sample['x']], y_train=sample['y'])

            f1, _, __ = model.evaluate(X_test=X_val, y_test=y_val, toFile=False)
            if f1 > best_f1:
                best_f1 = f1
                best_alpha_f1 = alpha
                best_beta_f1 = beta
                best_model_based_on_f1 = model
                best_m = m

print("Best model based on F1:")
best_model_based_on_f1.evaluate(X_test=X_test, y_test=y_test)
best_model_based_on_f1.save('MBW-F1.pkl')