import optuna
import subprocess
import os
import sys
import argparse
import importlib.util
import ipdb
def objective(trial):
    latent_size = trial.suggest_int('latent_size', 64, 128, step=8)  # 正确使用关键字参数
    i_latent_size = trial.suggest_int('i_latent_size', 4, 16, step=2)
    lr = trial.suggest_float('lr', 1e-5, 1e-4, log=True)  # 对学习率使用对数尺度搜索
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])  # 常见batch size选择
    alpha_p = trial.suggest_float('alpha_p', 0.0, 1.0)
    # 固定参数
    fixed_args = {
        'num_classes': 60,
        'ss': 5,
        'st': 'r',
        've': 'shift',
        'le': 'clip-vit-b-32',
        'tm': 'chat',
        'num_cycles': 10,
        'num_epoch_per_cycle': 1700,
        'phase': 'train',
        'mode': 'train',
        'dataset_path': '/usr1/home/s124mdg53_04/STAR/packed_features/shift_ntu60_5_r_2part',
        'wdir': 'char_results/shift_ntu60_5_r/',
        'dis_step': 4,
        'dataset': 'ntu60'
    }
    
    # 设置当前试验的工作目录
    trial_wdir = f"{fixed_args['wdir']}/trial_{trial.number}"
    os.makedirs(trial_wdir, exist_ok=True)
    
    # 准备命令行参数
    sys_argv_backup = sys.argv.copy()  # 备份当前的sys.argv
    
    # 构建新的参数列表
    sys.argv = ["train_char_align_score.py"]
    
    # 添加所有参数
    for key, value in fixed_args.items():
        if key == 'wdir':
            sys.argv.extend([f"--{key}", trial_wdir])
        else:
            sys.argv.extend([f"--{key}", str(value)])

    # 添加试验参数
    sys.argv.extend(["--latent_size", str(latent_size)])
    sys.argv.extend(["--i_latent_size", str(i_latent_size)])
    sys.argv.extend(["--lr", str(lr)])
    sys.argv.extend(["--batch_size", str(batch_size)])
    sys.argv.extend(["--alpha_p", str(alpha_p)])
    
    # 保存日志文件的路径
    log_file = f"{trial_wdir}/train.log"
    
    try:
        # 导入训练脚本模块
        spec = importlib.util.spec_from_file_location("train_module", "train_char_align_score.py")
        train_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(train_module)
        
        # 重定向stdout和stderr到日志文件
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        with open(log_file, 'w') as f:
            sys.stdout = f
            sys.stderr = f
            
            # 调用训练脚本的main函数，并获取返回的最佳准确率
            best_acc = train_module.main()
            
        # 恢复stdout和stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        
        print(f"Trial {trial.number} finished with best accuracy: {best_acc:.4f}")
        return best_acc
        
    except Exception as e:
        # 恢复stdout和stderr，以防在异常中被忽略
        if 'original_stdout' in locals():
            sys.stdout = original_stdout
        if 'original_stderr' in locals():
            sys.stderr = original_stderr
            
        print(f"Trial {trial.number} failed with error: {e}")
        return float('-inf')  # 返回一个很低的值，表示失败
    
    finally:
        # 恢复原始的sys.argv
        sys.argv = sys_argv_backup

def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter optimization")
    parser.add_argument('--n_trials', type=int, default=20, help='Number of trials')
    parser.add_argument('--study_name', type=str, default='char_align_optimization', help='Study name')
    parser.add_argument('--storage', type=str, default='sqlite:///char_align_optuna.db', help='Storage URL')
    parser.add_argument('--pruner', type=str, default='median', choices=['median', 'percentile', 'threshold', 'none'], help='Pruning algorithm')
    parser.add_argument('--n_jobs', type=int, default=1, help='Number of parallel jobs')
    args = parser.parse_args()
    
    # 设置pruner
    if args.pruner == 'median':
        pruner = optuna.pruners.MedianPruner()
    elif args.pruner == 'percentile':
        pruner = optuna.pruners.PercentilePruner(25.0)
    elif args.pruner == 'threshold':
        pruner = optuna.pruners.ThresholdPruner(0.2)
    else:
        pruner = optuna.pruners.NopPruner()
    
    # 创建或加载study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction='maximize',  # 最大化准确率
        pruner=pruner,
        load_if_exists=True
    )
    
    # 开始优化
    study.optimize(objective, n_trials=args.n_trials, n_jobs=args.n_jobs)
    
    # 输出最佳参数
    print("\n" + "="*60)
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (Accuracy): {trial.value:.4f}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # 保存最佳参数到文件
    with open('best_params.txt', 'w') as f:
        f.write(f"Best Accuracy: {trial.value:.4f}\n")
        for key, value in trial.params.items():
            f.write(f"{key}: {value}\n")
    
    print("\n=== 优化完成! 最佳参数已保存到 best_params.txt ===")
    print("你可以使用以下命令进行最终训练：")
    cmd = f"python train_char_align_score.py --num_classes 60 --ss 5 --st r --ve shift --le clip-vit-b-32 --tm chat --num_cycles 10 --num_epoch_per_cycle 1700 --latent_size {trial.params['latent_size']} --i_latent_size {trial.params['i_latent_size']} --lr {trial.params['lr']} --phase train --mode train --dataset_path \"/usr1/home/s124mdg53_04/STAR/packed_features/shift_ntu60_5_r_2part\" --wdir \"char_results/shift_ntu60_5_r/final\" --dis_step 4 --batch_size {trial.params['batch_size']} --dataset ntu60"
    print(cmd)

if __name__ == "__main__":
    main()