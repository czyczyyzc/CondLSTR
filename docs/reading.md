# 代码阅读

## 代码主体

   主要阅读`tools/train.py`文件，其包括了训练框架的各个组成部分:
    
### 参数解析部分
    
1. 输入参数解析:
    ```
    parser = argument_parser()
    main(parser.parse_args())
    ```
    
### 数据加载部分
    
1. 创建 data transformation:
    ```
    train_transforms, train_collate_fn = transforms.create(
        args.dataset, train=True, root=data_root, version=args.version)
    ```
2. 创建 dataset:
    ```
    train_dataset = datasets.create(
        args.dataset, data_root, split='train', transform=train_transforms, version=args.version)
    ```
3. 创建 dataloader:
    ```
    train_loader = dataloaders.create(
        args.dataset, train_dataset, args.batch_size, train=True, distributed=args.distributed,
        num_workers=args.num_workers, prefetch_factor=args.prefetch_factor, collate_fn=train_collate_fn)
    ```

 ### 模型创建部分

1. 创建 model:
    ```
    model = models.create(args.arch, norm_layer=norm_layer, model_path=args.model_path, accum_steps=args.accum_steps)
    ```

### 参数优化部分

1. 创建 optimizer:
    ```
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=tuple(args.betas), eps=args.eps,
                                    weight_decay=args.weight_decay)
    ```
2. 创建 scheduler:
    ```
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.milestones, gamma=args.gamma)
    ```
    
### 训练迭代部分

1. 创建 Trainer (负责一个epoch内的训练):
    ```
    trainer = Trainer(model, optimizer, scheduler, grad_scaler, autocast=autocast, max_grad=args.max_grad,
                        tb_writer=tb_writer, log_steps=args.log_steps, save_steps=args.save_steps,
                        accum_steps=args.accum_steps, distributed=args.distributed, root=args.logs_dir)
    ```

2. 进行多轮训练:
    ```
    for epoch in range(start_epoch, args.num_epochs):
        # Use .set_epoch() method to reshuffle the dataset partition at every iteration
        if hasattr(train_loader, 'set_epoch'):
            train_loader.set_epoch(epoch)

        if epoch >= args.train_epoch:
            trainer(train_loader, epoch, best_prec1)
    ```
    
## 代码展开
    
1. 每个模块都可以通过该模块对应的文件夹中的`__init__.py`文件中的`create`函数创建，这些模块包括:
    ```
    train_transforms, train_dataset, train_loader, model, ...
    ```
    可以跳转到`create`函数的定义处进行理解。

2. 以model的创建为例，该模块对应的文件夹为`modeling/models/models/`，该文件夹下的`__init__.py`文件内容包括：
    ```
    from .sam_vit import SamViT
    from .sam_hq_vit import SamHQViT
    from .sam_clip_mix import SamClipMix

    __factory = {
        'SamViT':             SamViT,
        'SamHQViT':           SamHQViT,
        'SamClipMix':         SamClipMix,
    }

    def create(name, *args, **kwargs):
        """
        Create a model instance.

        Parameters
        ----------
        name : str
            Model name. Can be one of 'inception', 'resnet18', 'resnet34',
            'resnet50', 'resnet101', and 'resnet152'.
        """
        if name not in __factory:
            raise KeyError("Unknown model:", name)
        return __factory[name](*args, **kwargs)
    ```

    其中，`__factory`对以一个字典的形式，对所有模型进行了注册，在训练时只要通过`-a`参数指定模型名称，就可以选择调用哪个模型，比如：
    ```
    python tools/train.py -a SamHQViT -d custom -t segmentation --data-dir /path/to/datasets/  --logs-dir /path/to/checkpoint -b 1 -j 4 -p amp --accum-steps 2 --model-path /path/to/pretrained/
    ```
    
    接着，`create`函数会把其余参数传入到指定模型的`__init__`函数中，从而对模型进行初始化：
    ```
    model = models.create(args.arch, norm_layer=norm_layer, model_path=args.model_path, accum_steps=args.accum_steps)
    ```

    具体的模型文件可以在文件夹`modeling/models/models/`下找到，比如`SamHQViT`的模型文件为`modeling/models/models/sam_hq_vit.py`.

3. 所有模块的创建（比如dataset, dataloader等）均和model的创建类似，都可以在该模块对应的文件夹中的`__init__.py`文件中，找到该模块的定义，都可以跳转到该模块定义的地方进行理解。
