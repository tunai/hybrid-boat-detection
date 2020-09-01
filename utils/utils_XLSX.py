import cv2
import xlsxwriter
from datetime import datetime


def createXLSX(site, name = None):

    # function to create/update anxlsx file

    now = datetime.now()
    now = now.strftime("%d-%m-%Y %H_%M_%S")
    if name is None:
        title = "[DETECTION SESSION " + site + '] ' + str(now) + '.xlsx'
    else:
        title = "[DETECTION SESSION " + site + " " + name + '] ' + str(now) + '.xlsx'
    workbook = xlsxwriter.Workbook(title)
    worksheet = workbook.add_worksheet()
    bold = workbook.add_format({'bold': True})
    worksheet.set_column(0, 0, 60)
    worksheet.set_column(1, 1, 35)
    worksheet.set_column(2, 2, 10)
    worksheet.set_column(3, 3, 10)
    worksheet.set_column(4, 4, 40)
    worksheet.set_column(5, 5, 10)

    worksheet.write(0, 0, "Address", bold)
    worksheet.write(0, 1, "UniqueID", bold)
    worksheet.write(0, 2, "Site", bold)
    worksheet.write(0, 3, "Num Boats", bold)
    worksheet.write(0, 4, "BBoxes", bold)
    worksheet.write(0, 5, "Scores", bold)
    return workbook, worksheet

def updateXLSX(worksheet, row, directory, names, site, numBoats, det, scores):

    worksheet.write(row, 0, directory+names)
    worksheet.write(row, 1, names)
    worksheet.write(row, 2, site)
    worksheet.write(row, 3, numBoats)
    worksheet.write(row, 4, det)
    worksheet.write(row, 5, scores)

    return worksheet
