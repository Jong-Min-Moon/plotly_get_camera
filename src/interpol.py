import string

def make_color_tuple( color ):
    """
    turn something like "#000000" into 0,0,0
    or "#FFFFFF into "255,255,255"
    """
    R = color[1:3]
    G = color[3:5]
    B = color[5:7]

    R = int(R, 16)
    G = int(G, 16)
    B = int(B, 16)

    return R,G,B

def interpolate_tuple( startcolor, goalcolor, steps ):
    """
    Take two RGB color sets and mix them over a specified number of steps.  Return the list
    """
    # white

    R = startcolor[0]
    G = startcolor[1]
    B = startcolor[2]

    targetR = goalcolor[0]
    targetG = goalcolor[1]
    targetB = goalcolor[2]

    DiffR = targetR - R
    DiffG = targetG - G
    DiffB = targetB - B

    buffer = []

    for i in range(0, steps +1):
        iR = int(R + (DiffR * i / steps))
        iG = int(G + (DiffG * i / steps))
        iB = int(B + (DiffB * i / steps))


        hR = hex(iR).replace("0x", "")
        hG = hex(iG).replace("0x", "")
        hB = hex(iB).replace("0x", "")

        if len(hR) == 1:
            hR = "0" + hR
        if len(hB) == 1:
            hB = "0" + hB

        if len(hG) == 1:
            hG = "0" + hG

        color = ("#"+hR+hG+hB).upper()
        buffer.append(color)

    return buffer

def interpolate( startcolor, goalcolor, steps ):
    """
    wrapper for interpolate_tuple that accepts colors as html ("#CCCCC" and such)
    """
    start_tuple = make_color_tuple(startcolor)
    goal_tuple = make_color_tuple(goalcolor)

    return interpolate_tuple(start_tuple, goal_tuple, steps)



def printchart(startcolor, endcolor, steps):

    colors = interpolate(startcolor, endcolor, steps)

    for color in colors:
        print(color)


# Example... show us 16 values of gradation between these two colors
