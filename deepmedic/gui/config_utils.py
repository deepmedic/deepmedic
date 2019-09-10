from PySide2 import QtWidgets, QtCore


def p(x):
    print(x)


def file_open(parent, dest, text='Search'):
    name, _ = QtWidgets.QFileDialog.getOpenFileName(parent, text)
    dest.setText(name)


def files_open(parent, dest, text='Search'):
    name, _ = QtWidgets.QFileDialog.getOpenFileNames(parent, text)
    dest.setText(str(name))


def folder_open(parent, dest, text='Search'):
    name = QtWidgets.QFileDialog.getExistingDirectory(parent, text)
    dest.setText(name)


def get_search_function(search_type='File'):
    if search_type == 'File':
        return file_open
    if search_type == 'Files':
        return files_open
    if search_type == 'Folder':
        return folder_open
    return None


def clear_all_texts(all_texts):
    [text.setText('') for text in all_texts]


def clear_all_comboboxes(all_comboboxes):
    [combobox.setCurrentIndex(0) for combobox in all_comboboxes]


def clear_all_checkboxes(all_checkboxes):
    [checkbox.setChecked(False) for checkbox in all_checkboxes]


def enable_on_combobox_value(combobox, text, to_enable):
    set_value = bool(str(combobox.currentText()) == text)
    for element in to_enable:
        element.setEnabled(set_value)


def get_widget_type(elem_type):
    if elem_type == 'lineedit':
        return QtWidgets.QLineEdit
    if elem_type == 'checkbox':
        return QtWidgets.QCheckBox
    if elem_type == 'combobox':
        return QtWidgets.QComboBox
    if elem_type == 'multiple':
        return QtWidgets.QLabel


def num(s):
    # returns a numeric value of the input s (if exists).
    # Examples: num('5') = 5
    #           num('[3, 4.5]') = [3, 4.5]
    #           num('[3, 'string']') = [3, 'string']
    #           num('string') = 'string'
    if s is None or s == "":
        return None
    if type(s) is bool:
        if s == 'False':
            return False
        else:
            return True
    if type(s) is dict:
        return s
    s = str(s)
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            is_tuple = False
            if s[0] == '(':
                is_tuple = True
            if s[0] == '[' or is_tuple:
                s = s[1:-1].split(',')
                s_len = len(s)
                for i in range(s_len):
                    s[i] = num(s[i])
                if is_tuple:
                    s = tuple(s)
                return s

            return s.strip().strip('"').strip("'")
