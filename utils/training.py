import matplotlib.pyplot as plt

class Trainer():
    def __init__(self, learn_required = 0.02, max_patience = 100):
        self.learn_required = learn_required
        self.max_patience = max_patience
        self.patience = 0
        self.state_dict = 0

    def __angry(self):
        self.patience += 1

    def __calm(self):
        self.patience = 0

    def __dissapoint(self):
        return self.max_patience <= self.patience
    
    def check(self, best, current):
        if best - current < current * self.learn_required:
            self.__angry()
        else:
            self.__calm()
        
        return self.__dissapoint()
    
    def comment(self):
        if self.patience > self.max_patience * 0.8:
            print("- Trainer: Model stops learning ... ({}/{})".format(self.patience, self.max_patience))
        elif self.patience > self.max_patience * 0.6:
            print("- Trainer: Model is really slowing down ({}/{})".format(self.patience, self.max_patience))
        elif self.patience > self.max_patience * 0.4:
            print("- Trainer: Model is not learning well ({}/{})".format(self.patience, self.max_patience))
        else:
            print("- Trainer: Model is in good shape ({}/{})".format(self.patience, self.max_patience))

class TrainRecord():
    def __init__(self, memory_len = 10):
        self.curr_epoch = 0
        self.curr_batch = 0
        self.start_epoch = 0
        self.pre_loss = 0
        self.curr_loss = 0
        self.losses = []
        self.recent_losses = [10 for i in range(memory_len)]
        self.avg_losses = []
        self.best_avg_loss = 10
        self.curr_avg_loss = 0
        self.memory_len = memory_len
        self.state_dict = 0
        self.is_new = True
        self.plot_figure = plt.figure()
        self.num_plots = 0

    def reset_plot(self):
        self.losses = []
        self.avg_losses = []
        self.is_new = True
    
    def record(self, curr_epoch, curr_batch, loss):
        if self.is_new:
            self.is_new = False
            self.start_epoch = curr_epoch

        self.curr_epoch = curr_epoch
        self.curr_batch = curr_batch

        self.pre_loss = self.curr_loss
        self.curr_loss = loss
        self.losses.append(loss)

        self.recent_losses.append(loss)
        self.recent_losses.pop(0)

        self.curr_avg_loss = sum(self.recent_losses) / self.memory_len            
        self.avg_losses.append(self.best_avg_loss)

    def update_best_avg_loss(self):
        if self.best_avg_loss > self.curr_avg_loss:
            self.best_avg_loss = self.curr_avg_loss
                
    def record_state_dict(self, state_dict):
        if self.pre_loss > self.curr_loss:
            self.state_dict = state_dict

    def add_subplot(self):
        new_axis = self.plot_figure.add_axes([0.1, 0.1 + 0.3 * self.num_plots, 0.5, 0.2])
        new_axis.plot(self.losses)
        new_axis.plot(self.avg_losses)
        self.num_plots += 1
        # plt.xlabel = "Epoch {} to {}".format(self.start_epoch, self.curr_epoch)
        # plt.plot(self.losses)
        # plt.plot(self.avg_losses)
        # plt.show()

        self.reset_plot()

    def plot(self):
        plt.show()


    def report(self, num_epochs, num_batches):
        print("Epoch {}/{}, Batch {}/{}: Loss = {}, Best average loss = {}".format(self.curr_epoch + 1, num_epochs, self.curr_batch + 1, num_batches, self.curr_loss, self.best_avg_loss))
