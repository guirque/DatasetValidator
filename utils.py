from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import colormaps
import os
import numpy as np

# Getting Color Pallete (remember to map it with dressings)
# https://matplotlib.org/stable/gallery/color/individual_colors_from_cmap.html
colors = colormaps.get_cmap('tab10').colors

name_mapping = {
 'acidos_graxos_essenciais_age': 'Ácidos Graxos Essenciais (AGE)',
 'adesivo_de_hidropolimero_espuma': 'Adesivo de Hidropolímero Espuma',
 'alginato_de_calcio': 'Alginato de Cálcio',
 'carvao_ativado_com_prata': 'Carvão Ativado com Prata',
 'colagenase': 'Colagenase',
 'filme_adesivo_transparente': 'Filme Adesivo Transparente',
 'hidrocoloide': 'Hidrocolóide',
 'hidrofibra_com_prata': 'Hidrofibra com Prata',
 'hidrogel': 'Hidrogel',
 'nylon_nao_aderente_impregnado_com_prata': 'Nylon não Aderente Impregnado com Prata',
 'sulfadiazina_de_prata': 'Sulfadiazina de Prata',
 'papaina_2_a_4_concetracao': 'Papaína com 2 a 4% de Concentração',
 'papaina_4_a_6_concetracao': 'Papaína com 4 a 6% de Concentração',
 'papaina_8_a_10_concetracao': 'Papaína com 8 a 10% de Concentração',
 'terapia_com_pressao_negativa': 'Terapia com Pressão Negativa',
 'acidos_graxos_essenciais_age_e_gaze': 'Ácidos Graxos Essenciais (AGE) + Gaze',
 'acidos_graxos_essenciais_age_e_filme_adesivo_transparente': 'Ácidos Graxos Essenciais (AGE) + Filme Adesivo Transparente',
 'alginato_de_calcio_e_gaze': 'Alginato de Cálcio + Gaze',
 'alginato_de_calcio_e_filme_adesivo_transparente': 'Alginato de Cálcio + Filme Adesivo Transparente',
 'alginato_de_calcio_e_espuma_adesiva': 'Alginato de Cálcio + Espuma Adesiva',
 'carvao_ativado_com_prata_e_gaze': 'Carvão Ativado com Prata + Gaze',
 'carvao_ativado_com_prata_e_filme_adesivo_transparente': 'Carvão Ativado com Prata + Filme Adesivo Transparente',
 'colagenase_e_gaze': 'Colagenase + Gaze',
 'colagenase_e_filme_adesivo_transparente': 'Colagenase + Filme Adesivo Transparente',
 'hidrofibra_com_prata_e_gaze': 'Hidrofibra com Prata + Gaze',
 'hidrofibra_com_prata_e_filme_adesivo_transparente': 'Hidrofibra com Prata + Filme Adesivo Transparente',
 'hidrofibra_com_prata_e_espuma_adesiva': 'Hidrofibra com Prata + Espuma Adesiva',
 'hidrogel_e_gaze': 'Hidrogel + Gaze',
 'hidrogel_e_filme_adesivo_transparente': 'Hidrogel + Filme Adesivo Transparente',
 'hidrogel_e_espuma_adesiva': 'Hidrogel + Espuma Adesiva',
 'nylon_nao_aderente_impregnado_com_prata_e_gaze': 'Nylon não Aderente Impregnado com Prata + Gaze',
 'nylon_nao_aderente_impregnado_com_prata_e_filme_adesivo_transparente': 'Nylon não Aderente Impregnado com Prata + Filme Adesivo Transparente',
 'sulfadiazina_de_prata_e_gaze': 'Sulfadiazina de Prata + Gaze',
 'sulfadiazina_de_prata_e_filme_adesivo_transparente': 'Sulfadiazina de Prata + Filme Adesivo Transparente',
 'papaina_2_a_4_concetracao_e_gaze': 'Papaína com 2 a 4% de Concentração + Gaze',
 'papaina_2_a_4_concetracao_e_filme_adesivo_transparente': 'Papaína com 2 a 4% de Concentração + Filme Adesivo Transparente',
 'papaina_4_a_6_concetracao_e_gaze': 'Papaína com 4 a 6% de Concentração + Gaze',
 'papaina_4_a_6_concetracao_e_filme_adesivo_transparente': 'Papaína com 4 a 6% de Concentração + Filme Adesivo Transparente',
 'papaina_8_a_10_concetracao_e_gaze': 'Papaína com 8 a 10% de Concentração + Gaze',
 'papaina_8_a_10_concetracao_e_filme_adesivo_transparente': 'Papaína com 8 a 10% de Concentração + Filme Adesivo Transparente',
 'NA': 'É possível que a lesão não seja por pressão, fugindo do contexto do modelo (NA: Not Applicable).',
 'ND': 'Foi escolhido não aplicar nenhuma cobertura.'
}


# https://medium.com/analytics-vidhya/iou-intersection-over-union-705a39e7acef, with change by vamsiikrishna (comment on post)
def IOU(bbox1, bbox2, img_size):
    # Coordinates
    c1, x1, y1, w1, h1 = bbox1
    c2, x2, y2, w2, h2 = bbox2

    img_w, img_h = img_size

    # Unnormalized Coordinates
    x1 *= img_w
    x2 *= img_w
    y1 *= img_h
    y2 *= img_h
    w1 *= img_w
    w2 *= img_w
    h1 *= img_h
    h2 *= img_h

    # Uncentralized Coordinates
    x1 -= w1/2
    y1 -= h1/2
    x2 -= w2/2
    y2 -= h2/2


    # intersection
    intersection_top_left = (max(x1, x2), max(y1, y2)) # top left corner more to the right gives us our x; top left corner more to the bottom gives us our y
    intersection_bottom_right = (min(x1+w1, x2+w2), min(y1+h1, y2+h2)) # bottom right corner more to the left gives us our x; highest bottom right gives us our y.

    intersection = max(0, intersection_bottom_right[0] - intersection_top_left[0]) * max(0, intersection_bottom_right[1] - intersection_top_left[1]) # w * h
    
    # union
    union = w1*h1 + w2*h2 - intersection

    return intersection/union if union > 0 else 0


def display_img_boxes(pil_img, labels, class_map, save_path, display_img=True, generate_img=True, output_file_prefix='saved_'):
    """
    Displays image with corresponding bounding boxes.

    # Arguments

    - *pil_img*: pil image to display.
    - *labels*: numpy matrix, with each row containing [label, x, y, w, h], with numbers normalized relative to image size.
    - *class_map*: list of class names (strings). The index represents the class id. 
    """

    mapping = []
    groups = np.zeros(len(labels)) # label of index i belongs to group groups[i]. group 0 is 'no group'
    group_index = 0

    # For each bounding box
    i = 0
    for label in labels:
        # Create subplots
        if generate_img:
            plt.clf()
            figure, axes = plt.subplots()
            axes.add_image(plt.imshow(pil_img))

        img_w, img_h = (pil_img.width, pil_img.height) # numpy shape will return [c, h, w]

        # Compare with all the next bounding boxes to verify the need for a group (compare with one_self to create a new group)
        for j in range(i, len(labels)):
            iou_score = IOU(labels[i], labels[j], (img_w, img_h))

            # Group them in OR statement
            if iou_score > 0.8:
                if groups[i] != 0:
                    groups[j] = groups[i]
                elif groups[j] != 0:
                    groups[i] = groups[j]
                else:
                    group_index += 1
                    groups[i] = groups[j] = group_index

            #print(f'IOU between {i} and {j}: {iou_score}')


        # https://docs.ultralytics.com/pt/datasets/detect/#ultralytics-yolo-format
        # values are normalized, so we need to "unnormalize" it
        # normalization: x <- px_x_center/img_width, w <- px_width/img_width,  y <- px_y_center/img_height, h <- px_height / img_height
        
        # boxes must be centralized! Matplotlib's Rectangle isn't like that by default (it defaults to the top left corner)
        # (coordinates are xy are treated as a starting point instead).
        # to centralize it, we'll: 
        # - On the x axis, we'll move it half width to the left
        # - On the y axis, we'll move it half height downwards

        # Label ------------
        class_label = class_map[int(label[0])]
        class_color = colors[int(label[0] % len(colors))]

        # Width, Height ----
        w, h = (label[3]*img_w, label[4]*img_h)

        if generate_img:
            # Coordinates ------
            unnormalized_coordinates = (label[1]*img_w, label[2]*img_h)
            uncentralized_coordinates = (unnormalized_coordinates[0] - w/2, unnormalized_coordinates[1] - h/2) # yolo coordinates are centralized, so we uncentralize them for matplotlib
            
            # Adding Rectangle and Corresponding Text =
            axes.add_patch(Rectangle(uncentralized_coordinates, w, h, fill=False, edgecolor='red', lw=2))
            plt.text(uncentralized_coordinates[0]+4, uncentralized_coordinates[1]-10, f'Área Destacada', color='black', weight='bold', backgroundcolor=(1, 1, 1, 0.4))

        i += 1
        dest_path = os.path.join(save_path, f'{output_file_prefix}{i}')
        mapping.append((
            dest_path+'.png', # path
            name_mapping[class_label], # human-readable label
            class_label, # actual label
            [label[i] for i in range(1, len(label))], # actual coordinates
        ))

        # https://stackoverflow.com/questions/8218608/savefig-without-frames-axes-only-content
        if generate_img:
            plt.axis('off')
            plt.savefig(dest_path, bbox_inches="tight", pad_inches=0.0, transparent=True)

            if display_img:
                print('Showing Image For Label: ', name_mapping[class_label])
                plt.show()
            plt.close()
    return mapping, groups

    # Other References
    # https://brandonrohrer.com/matplotlib_patches.html
    # https://matplotlib.org/stable/gallery/images_contours_and_fields/image_demo.html
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Rectangle.html
    # https://matplotlib.org/stable/users/explain/colors/colormaps.html