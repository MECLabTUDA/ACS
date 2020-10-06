import matplotlib.patches as mpatches

def _remove_empyties_and_duplicates(handles, labels, titles):
    '''Removes repeated entries and titles which are not followed by entries'''
    new_labels = []
    new_handles = []
    appeared = set()
    for ix, label in enumerate(labels):
        if label in appeared:
            continue
        else:
            appeared.add(label)
            if label in titles:
                # label is the last entry or the next label is a title
                if ix == len(labels) - 1 or labels[ix+1] in titles:
                    titles.remove(label)
                    continue
            new_labels.append(label)
            new_handles.append(handles[ix])
    return new_handles, new_labels, titles

def _bold_titles(labels, titles):
    '''
    Styles title labels bold
    '''
    labels = ['$\\bf{'+label+'}$' if label in titles else label for label in labels]
    titles = ['$\\bf{'+title+'}$' for title in titles]
    return labels, titles

def _insert_divider_before_titles(handles, labels, titles):
    '''
    Inserts an empty line before each new legend easthetic
    param titles: elements of 'labels' before which a space should be inserted
    '''
    titles = titles[1:] # Do not need to insert space before first title
    empty_handle = mpatches.Patch(color='white', alpha=0)
    space_indexes = [labels.index(title) for title in titles]
    for i in range(len(space_indexes)):
        handles.insert(space_indexes[i], empty_handle)
        labels.insert(space_indexes[i], '')
        space_indexes = [i+1 for i in space_indexes]
    return handles, labels

def _add_hue_dimension(handles, labels):
    # TODO
    handles.append(mpatches.Patch(color='red', alpha=0.5))
    labels.append('white')
    return handles, labels

def format_legend(ax, titles):
    if 'numpy' in str(type(ax)):
        ax = ax.copy()[-1]
    # Fetch legend labels and handles
    handles, labels = ax.get_legend_handles_labels()
    handles, labels, titles = _remove_empyties_and_duplicates(handles, labels, titles)
    labels, titles = _bold_titles(labels, titles)
    handles, labels = _insert_divider_before_titles(handles, labels, titles)
    # Legend to the side
    ax.legend(handles, labels, bbox_to_anchor=(1, 1), loc=2)

def _add_training_legend_items(handles, labels, alpha_training, alpha_not_training):
    handles.append(mpatches.Patch(color='white', alpha=0))
    handles.append(mpatches.Patch(color='black', alpha=alpha_training))
    handles.append(mpatches.Patch(color='black', alpha=alpha_not_training))
    labels.append('Training')
    labels.append('On data')
    labels.append('On other data')
    return handles, labels