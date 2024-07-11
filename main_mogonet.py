from train_test import train_test
# 主函数
if __name__ == "__main__":
    # ROSMAP用于阿尔茨海默病(AD)患者与正常对照组(NC)的分类,BRCA用于乳腺浸润性癌(BRCA) PAM50亚型分类
    # LGG胶质瘤分级, KIPAN肾癌类型分类
    data_folder = 'ROSMAP'
    view_list = [1,2,3]

    num_epoch_pretrain = 500
    num_epoch = 2500
    lr_e_pretrain = 1e-3
    lr_e = 5e-4
    lr_c = 1e-3

    theta_smooth = 1
    theta_degree = 0.8
    theta_sparsity = 0.5
    reg = 0.001
    neta = 0.1
    
    if data_folder == 'ROSMAP':
        num_class = 2
    if data_folder == 'BRCA':
        num_class = 5
    if data_folder == 'KIPAN':
        num_class = 3
    if data_folder == 'LGG':
        num_class = 2

    train_test(data_folder, view_list, num_class,
               lr_e_pretrain, lr_e, lr_c, 
               num_epoch_pretrain, num_epoch, theta_smooth, theta_degree, theta_sparsity, neta, reg)