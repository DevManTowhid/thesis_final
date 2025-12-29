import torch
import os
import re
import json
import valid
from utils import utils
from utils import option
from data import dataset
from model import HTR_VT
from collections import OrderedDict
from tqdm import tqdm  # [1] Import tqdm

def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    args.save_dir = os.path.join(args.out_dir, args.exp_name)
    os.makedirs(args.save_dir, exist_ok=True)
    logger = utils.get_logger(args.save_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

    model = HTR_VT.create_model(nb_cls=args.nb_cls, img_size=args.img_size[::-1])

    # --- MODIFIED: Load based on the defined 'total_iter' ---
    # 1. Construct the path to the specific iteration folder (MATCHING TRAIN.PY LOGIC)
    # We reconstruct the folder name exactly as train.py does:
    folder_name = f"{args.exp_name}_{args.subcommand}_{args.total_iter}_{args.train_bs}"
    iter_path = os.path.join(args.save_dir, f'iter_{folder_name}', 'best_CER.pth')
   
    
    # 2. Logic: Check if that specific folder exists
    if os.path.exists(iter_path):
        pth_path = iter_path
        logger.info(f'📍 Found specific iteration checkpoint: {pth_path}')
    else:
        # Fallback: Try the main folder if the specific iter folder is missing
        logger.info(f'⚠️ Folder "iter_{args.total_iter}" not found. Checking default location...')
        pth_path = os.path.join(args.save_dir, 'best_CER.pth')

    if not os.path.exists(pth_path):
        logger.error(f"❌ Error: No checkpoint found at {pth_path}")
        return
    # ---------------------------------------------------------

    logger.info('loading HWR checkpoint from {}'.format(pth_path))

    ckpt = torch.load(pth_path, map_location='cpu')
    model_dict = OrderedDict()
    pattern = re.compile('module.')

    for k, v in ckpt['state_dict_ema'].items():
        if re.search("module", k):
            model_dict[re.sub(pattern, '', k)] = v
        else:
            model_dict[k] = v

    model.load_state_dict(model_dict, strict=True)
    model = model.cuda()

    logger.info('Loading test loader...')
    train_dataset = dataset.myLoadDS(args.train_data_list, args.data_path, args.img_size)

    test_dataset = dataset.myLoadDS(args.test_data_list, args.data_path, args.img_size, ralph=train_dataset.ralph)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.val_bs,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=args.num_workers)

    converter = utils.CTCLabelConverter(train_dataset.ralph.values())
    criterion = torch.nn.CTCLoss(reduction='none', zero_infinity=True).to(device)

    model.eval()
    with torch.no_grad():
        # [2] Wrap test_loader with tqdm for the progress bar
        val_loss, val_cer, val_wer, preds, labels = valid.validation(
            model,
            criterion,
            tqdm(test_loader, desc="Evaluating", unit="batch"), 
            converter
        )

    logger.info(
        f'Test. loss : {val_loss:0.3f} \t CER : {val_cer:0.4f} \t WER : {val_wer:0.4f} ')


if __name__ == '__main__':
    args = option.get_args_parser()
    main()
