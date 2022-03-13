"""
Author: Rayla Kurosaki

File: my_colors.py

Description:
"""


def get_math_colors():
    black = "#000000"
    red = "#FF0000"
    blue = "#0000FF"
    return [black, red, blue]


def get_shades_cyan():
    aqua = "#00FFFF"
    celeste = "#B2FFFF"
    electric_blue = "#7DF9FF"
    sky_blue = "#80DAEB"
    turquoise = "#40E0D0"
    return [aqua, celeste, electric_blue, sky_blue, turquoise]


def get_shades_pink():
    pink = "#FFC0CB"
    hot_pink = "#FF69B4"
    deep_pink = "#FF1493"
    pink_lace = "#FFDDF4"
    cherry_blossom_pink = "#FFB7C5"
    light_hot_pink = "#FFB3DE"
    lavender_pink = "#FBAED2"
    rose_pink = "#FF66CC"
    light_deep_pink = "#FF5CCD"
    ultra_pink = "#FF6FFF"
    shocking_pink = "#FC0FC0"
    steel_pink = "#CC33CC"
    return [pink, hot_pink, deep_pink, pink_lace, cherry_blossom_pink,
            light_hot_pink, lavender_pink, rose_pink, light_deep_pink,
            ultra_pink, shocking_pink, steel_pink]


def get_genshin_colors(color_type):
    match color_type:
        case "Rarity":
            return {4: "#64578D", 5: "#A1662A"}
        case "Elements":
            return {"Anemo": "#A4F6CC", "Geo": "#F1D85F",
                    "Electro": "#DABBFD", "Dendro": "#A7D43B",
                    "Hydro": "#06E9F9", "Pyro": "#FBAB71", "Cryo": "#A6FDFE"}
        case "Nation":
            return {"Mondstadt": "#A4F6CC", "Liyue": "#F1D85F",
                    "Inazuma": "#DABBFD", "Sumeru": "#A7D43B",
                    "Fontaine": "#06E9F9", "Natlan": "#FBAB71",
                    "Snezhnaya": "#A6FDFE"}
    return None


def get_red_yellow_green_colorscale():
    return ["#F8696B", "#FFEB84", "#63BE7B"]
