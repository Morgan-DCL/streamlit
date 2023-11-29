import pandas as pd
from tools_app import (
    afficher_details_film,
    afficher_top_genres,
    auto_scroll,
    clean_dup,
    get_clicked,
    get_info,
    infos_button,
    knn_algo,
)

import streamlit as st

# Configuration de la page
st.set_page_config(
    page_title="DigitalDreamers Recommandation System",
    page_icon="📽️",
    initial_sidebar_state="collapsed",
    layout="wide",
)
# Supprime les boutons fullscreen des images de l"app.
hide_img_fs = """
    <style>
    button[title="View fullscreen"]{
        visibility: hidden;
    }
    </style>
"""
st.markdown(hide_img_fs, unsafe_allow_html=True)
# Arrondi les coins des images.
round_corners = """
    <style>
        .st-emotion-cache-1v0mbdj > img{
            border-radius:2%;
        }
    </style>
"""
st.markdown(round_corners, unsafe_allow_html=True)
st.markdown(
    """
    <script>
        function streamlit_on_click(index) {
            const buttonCallbackManager = Streamlit.ButtonCallbackManager.getManager();
            buttonCallbackManager.setCallback("infos_button", index => {
                Streamlit.setComponentValue(index);
            });
            buttonCallbackManager.triggerCallback("infos_button", index);
        }
    </script>
    """,
    unsafe_allow_html=True,
)

# Importation des dataframes nécessaires.
machine_learning = "datasets/machine_learning_final.parquet"
site_web = "datasets/site_web.parquet"
df_ml = pd.read_parquet(machine_learning)
df_ml = clean_dup(df_ml)
df_sw = pd.read_parquet(site_web)
df_sw = clean_dup(df_sw)

# Création de la liste des films pour la sélection.
default_message = "Entrez ou sélectionnez le nom d'un film..."
movies = df_sw["titre_str"]
movies_list = [default_message] + list(sorted(movies))
selectvalue = default_message


def callback():
    st.session_state["button_clicked"] = True


def callback2():
    st.session_state["button_clicked"] = False


# Début de la page.
st.session_state["clicked"] = None
st.header("DigitalDreamers Recommandation System", anchor=False)
# Instanciation des session_state.
if "index_movie_selected" not in st.session_state:
    st.session_state["index_movie_selected"] = movies_list.index(
        selectvalue
    )
if "clicked" not in st.session_state:
    st.session_state["clicked"] = None
if "counter" not in st.session_state:
    st.session_state["counter"] = 1
if "button_clicked" not in st.session_state:
    st.session_state["button_clicked"] = False

# # TESTING POUR ALLER SUR LA PAGE full_bio.py quand je clique sur l'image des acteurs et directors
# if 'page_actuelle' not in st.session_state:
#     st.session_state.page_actuelle = "principale"


# def page_principale():
#     st.title("Page Principale")
#     # Contenu de la page principale

# def autre_page():
#     st.title("Autre Page")
#     # Contenu de l'autre page

# if st.session_state.page_actuelle == "principale":
#     page_principale()

# elif st.session_state.page_actuelle == "autre":
#     autre_page()


top = 10

# Barre de sélection de films.
selectvalue = st.selectbox(
    label="Choisissez un film ⤵️",
    options=movies_list,
    placeholder=default_message,
    index=st.session_state["index_movie_selected"],
)
if selectvalue != default_message:
    selected_movie = df_sw[df_sw["titre_str"] == selectvalue]
    afficher_details_film(selected_movie)
    synop, recom = st.columns([3, 4])
    with synop:
        st.subheader("**Synopsis :**", anchor=False, divider=True)
        st.markdown(get_info(selected_movie, "overview"))
    with recom:
        st.subheader("**Films Similaires :**", anchor=False, divider=True)
        st.markdown("</div>", unsafe_allow_html=True)
        top = 5
        recommended = knn_algo(df_ml, selectvalue, top)
        cols = st.columns(top)
        for i, col in enumerate(cols):
            with col:
                index, clicked = get_clicked(df_sw, recommended, i)
                if clicked:
                    st.session_state["button_clicked"] = False
                    st.session_state["clicked"] = index
        if st.session_state["clicked"] is not None:
            infos_button(df_sw, movies_list, st.session_state["clicked"])
            st.session_state["counter"] += 1
            auto_scroll()
            st.rerun()
    auto_scroll()
else:
    st.markdown("<br>", unsafe_allow_html=True)
    st.write("Comment utiliser l'application de recommandations :")
    st.write("1. Choisissez ou entrer le nom d'un film.")
    st.write(
        "2. Cliquez sur le bouton en haut de l'écran pour voir les films similaires."
    )
    st.write(
        "3. Cliquez sur une des recommandations pour avoir plus d'infos."
    )
    st.markdown("<br><br>", unsafe_allow_html=True)

    genres_list = [
        "Drame",
        "Comédie",
        "Animation",
        "Action",
        "Romance",
        "Crime",
    ]
    for genre in genres_list:
        genre_df = afficher_top_genres(df_sw, genre)
        titres = genre_df["titre_str"].head(top).tolist()
        st.header(f"Top {top} {genre} :", anchor=False)
        cols = st.columns(top)
        for i, col in enumerate(cols):
            with col:
                index, clicked = get_clicked(
                    genre_df, titres, i, genre, True
                )
                if clicked:
                    st.session_state["clicked"] = index
        if st.session_state["clicked"] is not None:
            infos_button(df_sw, movies_list, st.session_state["clicked"])
            st.session_state["counter"] += 1
            auto_scroll()
            st.rerun()
    auto_scroll()

st.write(
    "App développée par [Morgan](https://github.com/Morgan-DCL) et [Teddy](https://github.com/dsteddy)"
)
