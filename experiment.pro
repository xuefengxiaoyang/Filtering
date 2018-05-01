QT += core
QT -= gui

TARGET = experiment
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

SOURCES += main.cpp

INCLUDEPATH += D:\OpencvQt\include \
               D:\OpencvQt\include\opencv \
               D:\OpencvQt\include\opencv2

LIBS += D:\OpencvQt\lib\libopencv_*
