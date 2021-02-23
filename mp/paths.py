# ------------------------------------------------------------------------------
# Module where paths should be defined.
# ------------------------------------------------------------------------------
import os

# Path where intermediate and final results are stored
storage_path = 'storage'
storage_data_path = os.path.join(storage_path, 'data')

# Original data paths. TODO: set necessary data paths.
# original_data_paths = {'example_dataset_name': 'storage/data'}
original_data_paths = {'DecathlonHippocampus': 'storage/data/DecathlonHippocampus',
                        'DryadHippocampus': 'storage/data/DryadHippocampus',
                        'HarP': 'storage/data/HarP'}

# Login for Telegram Bot
telegram_login = {'chat_id': 'TODO', 'token': 'TODO'}
