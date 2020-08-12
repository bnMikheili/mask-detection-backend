import yaml


def import_config_data_full():
    """
    Util function to import entire yaml file

    Returns:
        (dict): dictionary of all conf data
    """
    with open('config/config.yaml') as config_file:
        config_data = yaml.load(config_file, Loader=yaml.FullLoader)
    return config_data


def get_config_data_by_key(category):
    """
    Util function to import category of data

    Args:
        category (str): valid string in yaml file

    Returns:
        (dict): dictionary of conf data indexed by 'category'
    """
    data = import_config_data_full()
    return data.get(category)


def get_config_data_by_keys(categories):
    """
    Util function to import categories of data

    Args:
        categories (list): list of strings that are found in yaml file

    Returns:
        (dict): dictionary of conf data indexed by each category
    """
    data = {}
    for category in categories:
        data.update(get_config_data_by_key(category))
    return data
