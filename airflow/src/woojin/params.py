train_data_path = './data/cleansed/test.csv'
train_columns = ['Plasticizing_Time'] # ['Max_Switch_Over_Pressure', 'Cycle_Time', 'Max_Injection_Pressure', 'Barrel_Temperature_6'] # change column names if EDA result is changed
time_columns = "TimeStamp" # Update name if TimeStamp column name is changed

interval = 1
latent_dim = 20
shape = [100, 1]
encoder_input_shape = [100, 1]
generator_input_shape = [20, 1]
critic_x_input_shape = [100, 1]
critic_z_input_shape = [20, 1]
encoder_reshape_shape = [20, 1]
generator_reshape_shape = [50, 1]
learning_rate = 0.0005
batch_size = 64
n_critics = 5
epochs = 1000
check_point = 1
z_range =[0, 10]
window_size = None
window_size_portion = None
window_step_size = None
window_step_size_portion = None
min_percent = 0.1
anomaly_padding =100

usecols=['Injection_Time',
        'Filling_Time',
        'Plasticizing_Time',
        'Cycle_Time',
        'Clamp_Close_Time',
        'Cushion_Position',
        'Switch_Over_Position',
        'Clamp_Open_Position',
        'Max_Injection_Speed',
        'Max_Screw_RPM',
        'Average_Screw_RPM',
        'Max_Injection_Pressure',
        'Max_Switch_Over_Pressure',
        'Max_Back_Pressure',
        'Average_Back_Pressure',
        'Barrel_Temperature_2',
        'Barrel_Temperature_3',
        'Barrel_Temperature_4',
        'Barrel_Temperature_5',
        'Barrel_Temperature_6',
        'Barrel_Temperature_7']