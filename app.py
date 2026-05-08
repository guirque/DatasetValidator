import streamlit as st
import pandas as pd
import re
from utils import display_img_boxes
from PIL import Image
import os
import numpy as np
from xml.dom.minidom import parseString
from db import add_registers

login_data = {
    'lpp': st.secrets['user_data']['lpp']
}

data = {
    'lpp': {
        'dataset_path': 'dataset',
        'dataset_csv_path': 'dataset/final_dataset_postprocessed.csv',
        'temp_folder_path': 'temp'
    }
}


# How is data stored?
# data dict has fixed data.
# st.session_state['user_progress'] holds a dict with the feedback for each image they've registered. The key is the image name. The value is a dict containing relevant info.
# E.g.:
# {
#   "an_img.png": {
#       "approved": True,
#       "description": "Claro e aceitável.",
#       "adequate_choices": True,
#       "lang_error": False,
#       "author": "Username",
#       "img": "img_name.png"
#    }
# }

if 'user_progress' not in st.session_state.keys():
    st.session_state['user_progress'] = {}

# Main -----------------------------------------

def main(data_df):
    with st.container():
        save = st.button('Salvar Progresso', icon='💾', type='secondary')
        if save:
            objs = st.session_state['user_progress'].values() 
            add_registers(objs)
            st.write('✅ Salvo!')
    st.write(f'# Dataset "*{st.session_state["ds"]}*"')
    with st.expander(label='Feedback'):

        st.write('Escolha as melhores opções de acordo com o raciocínio e a escolha de coberturas associados à imagem. Lembre-se de considerar todas as áreas destacadas.')
        with st.form(key='feedback', border=False):
            st.text('Aprovada? ')
            approv = st.feedback()
            desc = st.selectbox(
                label='Qual descrição mais se encaixa com o raciocínio apresentado?',
                options=[
                    'Claro e aceitável.',
                    'Confuso ou contraditório.',
                    'Correto, mas incompleto.',
                    'Incorreto.'
                ]
            )
            adeq = st.checkbox(label='A escolha das coberturas foi adequada?')
            lang = st.checkbox(label='Há erros de idioma ou de termos técnicos?')
            submit = st.form_submit_button(label='Registrar')

            if submit and 'last_img' in st.session_state.keys():
                st.session_state['user_progress'][st.session_state['last_img']] = {
                    'approved': True if approv == 1 else False,
                    'description': desc,
                    'adequate_choices': adeq,
                    'lang_error': lang,
                    'author': st.session_state['username'],
                    'img': st.session_state['last_img']
                }
                st.write('✅ Registrado!')


    with st.container(vertical_alignment='center', width='stretch'):
        st.write('## Conteúdo')
        if 'main_slider' not in st.session_state.keys():
            st.session_state['main_slider'] = 0

        with st.container(border=True):
            st.write('Imagem Selecionada')
            with st.container(horizontal=True, horizontal_alignment='distribute', vertical_alignment='center'):
                def prev_func():
                    if st.session_state['main_slider'] > 0:
                        st.session_state['main_slider'] -= 1
                        st.write('prev')
                def next_func():
                    if st.session_state['main_slider'] < data_df.shape[0]-1:
                        st.session_state['main_slider'] += 1

                # https://discuss.streamlit.io/t/any-plan-to-support-the-value-of-sidebar-slider-update/16052/2
                st.button(label='<', on_click=prev_func)
                st.slider(label='image', label_visibility='collapsed', key='main_slider', min_value=0, max_value=data_df.shape[0]-1, bind='query-params')
                st.button(label='>', on_click=next_func)
    
        # Set variables
        row = data_df.iloc[st.session_state['main_slider']]
        DATASET_PATH = data[st.session_state['ds']]['dataset_path']
        IMAGES_PATH = os.path.join(DATASET_PATH, 'images')
        CLASSES_TXT_PATH = os.path.join(DATASET_PATH, 'classes.txt')
        LABELS_PATH = os.path.join(DATASET_PATH, 'labels')
        TEMP_IMAGES_FOLDER = data[st.session_state['ds']]['temp_folder_path']

        # Load BBoxes for the Image being Analyzed
        class_map_list = []

        with open(CLASSES_TXT_PATH, 'r') as file:
            class_map_list = file.read().strip().split('\n')

        # Reading text files
        img_name = row['img_path'].split('/')[-1]
        st.session_state['last_img'] = img_name
        image = Image.open(os.path.join(IMAGES_PATH, img_name))
        # Get bboxes from labels file
        txt_file = '.'.join(img_name.split('.')[:-1]) + '.txt'
        bboxes = []
        with open(os.path.join(LABELS_PATH, txt_file), 'r') as read_labels:
            file_content = read_labels.read()
            lines = file_content.split('\n')[:-1]
            for bbox in lines:
                values = [float(i) for i in bbox.split(' ')]
                bboxes.append(values)

        # Displaying
        if len(bboxes) != 0:
            results, groups = display_img_boxes(image, bboxes, class_map_list, TEMP_IMAGES_FOLDER, display_img=False)

            unique_groups = np.unique(np.array(groups), return_index=True)[1]

            bbox_index = 0
            if len(unique_groups) > 1:
                with st.container(border=True):
                    st.write('Caixa de Localização (BBox)')
                    bbox_index = st.slider(label='bounding box', label_visibility='collapsed', min_value=0, max_value=len(unique_groups)-1)

            group_index = unique_groups[bbox_index]
            group_index = int(group_index)

            chosen_region_num = int(groups[group_index]) # group (region) the bbox belongs to
            region_string = re.findall(r'<region>[\s\S]*?<\/region>', row['final_ai_response'])[chosen_region_num - 1]

            st.columns(3)[1].image(Image.open(results[group_index][0]), caption=img_name)
            for opt in re.findall(r'<opt>[\S\s]*?<\/opt>', region_string):
                try:
                    dom = parseString(opt)
                    treatment = dom.getElementsByTagName('treatment')[0].firstChild.nodeValue
                    reason = dom.getElementsByTagName('reason')[0].firstChild.nodeValue

                    st.write(f'### Tratamento: {treatment}')
                    st.write(f'{reason}')
                except:
                    st.write('**Atenção: esta resposta não está formatada corretamente.** Por isso, não foi possível extrair uma lista de tratamentos e raciocínios. É possível que ela deva ser rejeitada de imediato.')
                    st.write('Segue a resposta sem processamento:\n')
                    st.code(opt)
        else:
             st.image(image)
    



# Login ---------------------------------------------------------

def login():
    st.write('# Vali.ds')
    st.text('Digite a senha de acesso criada pelo admin para prosseguir.')
    ds_name = st.text_input(label='Nome do Dataset: ', type='default')
    username = st.text_input(label='Seu nome: ', type='default')
    password = st.text_input(label='Senha: ', type='password')
    login = st.button(label='Log In')

    # Auth
    if login and ds_name in login_data.keys() and password == login_data[ds_name]:
        st.session_state['ds'] = ds_name
        st.session_state['password'] = login_data[ds_name]
        st.session_state['username'] = username
        st.rerun()
    elif login:
        st.write('Dataset não existe ou senha incorreta.')

def auth_page():

    # If in session data
    if 'ds' in st.session_state.keys():
        ds_name = st.session_state['ds']
        password = st.session_state['password']

        if ds_name in login_data.keys() and password == login_data[ds_name]:
            df = pd.read_csv(data[ds_name]['dataset_csv_path'], index_col=None)
            main(df)
        else:
            login()
    else:
        login()
        
auth_page()