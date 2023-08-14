import sys
import math
import os
import numpy as np
from datetime import datetime, timedelta
import pathlib
import wget


from gnssVisual import (createSatellitesVisibleSatellitesPositions, plotSkyplotFromSatVisibleOnAxis, plot3dSatellitesOnAxis,
satellitesPositionsForWholePeriod2satellitesPhiLam, plotGroundtrackOnAxis, plotVisibleSatellitesNumberOnAxis, plotDOPsOnAxis,
plotElevationsOnAxis
)

from PyQt6.QtWidgets import (
    QApplication, QLineEdit, 
    QPushButton, QVBoxLayout, 
    QLabel, QGridLayout, 
    QFileDialog, QComboBox,  
    QLineEdit,
    QHBoxLayout,
    QFormLayout, QTabWidget,
    QWidget, QDateTimeEdit,
    QMessageBox, QCheckBox
    )
from PyQt6.QtGui import QDoubleValidator, QIntValidator
from PyQt6 import QtWidgets
from PyQt6.QtCore import QDateTime, Qt

from matplotlib.backends.backend_qt5agg import FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

class MyApp(QtWidgets.QMainWindow):
### High level
    def __init__(self):
        super().__init__()
        self.setVariables()
        self.setUI()

    def setVariables(self):
        self.winTitle = "Satellites positions"
        self.angleUnits = ["Degrees (°)", "Radians (rad)", "Grad"]
        self.lenghtUnits = ["Meters", "Feet"]
        self.almanacFilePath = ""
        self.minWidth, self.minHeight = 500, 500
        self.prns = []
        self.satellitesPositionsForWholePeriod = []
        self.satellitesVisibleOverWholePeriod = []
        self.matricesAAndDOPsOverWholePerid = []
        self.visualisationsPrnsDict = dict()
        self.visualisationsChecksDict = dict()
        self.visualisationsPrnsNoColumns = 10

    def setUI(self):
        self.setWindowTitle(self.winTitle)
        self.setMinimumSize(self.minWidth, self.minHeight)
        layout = QVBoxLayout()

        tabsWid = QTabWidget()
        tabsWid.setTabPosition(QTabWidget.TabPosition.South)
        tabsWid.addTab(self.inputTabUI(), "Input")
        tabsWid.addTab(self.sphereVisualisationTabUI(), "Sphere Vis.")
        tabsWid.addTab(self.skyPlotVisualisationTabUI(), "Skyplot Vis.")
        tabsWid.addTab(self.groundtrackTabUI(), "Groundtrack Vis.")
        tabsWid.addTab(self.numberOfVisibleSatellitesVisualisationTabUI(), "Number of Visible Satellites Vis.")
        tabsWid.addTab(self.DOPsTabUI(), "DOPs Vis.")
        tabsWid.addTab(self.elevationsTabUI(), "Elevations Vis.")

        self.tabsWid = tabsWid
        layout.addWidget(self.tabsWid)

        widget = QtWidgets.QWidget()
        widget.setLayout(layout)

        self.setCentralWidget(widget)
        self.show()

### UI
    def inputTabUI(self):
        inputTab = QWidget()
        self.inputLayout = QVBoxLayout()

        self.setGeneralInputs(self.inputLayout)
        
        tmpbtn = QPushButton("Clear")
        tmpbtn.clicked.connect(self.clear)
        self.inputLayout.addWidget(tmpbtn)

        tmpbtn = QPushButton("Download new almanac file")
        tmpbtn.clicked.connect(self.downloadAlmanac)
        self.inputLayout.addWidget(tmpbtn)

        tmpbtn = QPushButton("Check inputs")
        tmpbtn.clicked.connect(self.updateInputsStateLabel)
        self.inputLayout.addWidget(tmpbtn)

        tmpbtn = QPushButton("Calculate")
        tmpbtn.clicked.connect(self.calculate)
        self.inputLayout.addWidget(tmpbtn)

        self.labelInputState = QLabel("Inputs' state:")
        self.labelInputState.setAlignment(Qt.AlignmentFlag(4))
        self.inputLayout.addWidget(self.labelInputState)
        
        self.visualisationsGnssChoiceLayout = QGridLayout()
        self.inputLayout.addLayout(self.visualisationsGnssChoiceLayout)

        self.visualisationsPrnsDict = dict()
        self.visualisationsPrnsLayout = QGridLayout()
        self.inputLayout.addLayout(self.visualisationsPrnsLayout)
        inputTab.setLayout(self.inputLayout)
        return inputTab

    def downloadAlmanac(self):
        file = 'ftp://ftp.trimble.com/pub/eph/Almanac.alm'
        path = str(pathlib.Path(__file__).parent.resolve())+"/Almanac.alm"
        wget.download(file,path)

    def updatePrnsCheckboxesInInputTab(self):
        if all(self.checkCalculated()):
            if len(self.visualisationsChecksDict.keys()) == 0:
                self.visualisationsChecksDict = dict()
                self.visualisationsChecksDict["G"] = QCheckBox("GPS")
                self.visualisationsChecksDict["R"] = QCheckBox("Glonos")
                self.visualisationsChecksDict["E"] = QCheckBox("Galileo")
                self.visualisationsChecksDict["Q"] = QCheckBox("QZSS")
                self.visualisationsChecksDict["I"] = QCheckBox("IRNSS")
                self.visualisationsChecksDict["C"] = QCheckBox("Beidou")
                self.visualisationsChecksDict["S"] = QCheckBox("Other")

                self.visualisationsChecksDict["G"].stateChanged.connect(self.updatePrnsCheckboxesOnMainCheckboxStateChange)
                self.visualisationsChecksDict["R"].stateChanged.connect(self.updatePrnsCheckboxesOnMainCheckboxStateChange)
                self.visualisationsChecksDict["E"].stateChanged.connect(self.updatePrnsCheckboxesOnMainCheckboxStateChange)
                self.visualisationsChecksDict["Q"].stateChanged.connect(self.updatePrnsCheckboxesOnMainCheckboxStateChange)
                self.visualisationsChecksDict["I"].stateChanged.connect(self.updatePrnsCheckboxesOnMainCheckboxStateChange)
                self.visualisationsChecksDict["C"].stateChanged.connect(self.updatePrnsCheckboxesOnMainCheckboxStateChange)
                self.visualisationsChecksDict["S"].stateChanged.connect(self.updatePrnsCheckboxesOnMainCheckboxStateChange)
            
                for i, checkBox in enumerate(self.visualisationsChecksDict.values()):
                    self.visualisationsGnssChoiceLayout.addWidget(checkBox, 0, i)
        
            for i, prn in enumerate(self.prns):
                row = i // self.visualisationsPrnsNoColumns
                column = i % self.visualisationsPrnsNoColumns
                self.visualisationsPrnsDict[prn] = QCheckBox(prn)
                self.visualisationsPrnsLayout.addWidget(self.visualisationsPrnsDict[prn], row, column)

    def updatePrnsCheckboxesOnMainCheckboxStateChange(self):
        if not all(self.checkCalculated()):
            return False
        if self.visualisationsChecksDict["G"].isChecked():
            for key in [k for k in self.visualisationsPrnsDict.keys() if k[0] == "G"]:
                self.visualisationsPrnsDict[key].setChecked(True)
        else:
            for key in [k for k in self.visualisationsPrnsDict.keys() if k[0] == "G"]:
                self.visualisationsPrnsDict[key].setChecked(False)
                
        if self.visualisationsChecksDict["R"].isChecked():
            for key in [k for k in self.visualisationsPrnsDict.keys() if k[0] == "R"]:
                self.visualisationsPrnsDict[key].setChecked(True)
        else:
            for key in [k for k in self.visualisationsPrnsDict.keys() if k[0] == "R"]:
                self.visualisationsPrnsDict[key].setChecked(False)
                
        if self.visualisationsChecksDict["E"].isChecked():
            for key in [k for k in self.visualisationsPrnsDict.keys() if k[0] == "E"]:
                self.visualisationsPrnsDict[key].setChecked(True)
        else:
            for key in [k for k in self.visualisationsPrnsDict.keys() if k[0] == "E"]:
                self.visualisationsPrnsDict[key].setChecked(False)
                
        if self.visualisationsChecksDict["Q"].isChecked():
            for key in [k for k in self.visualisationsPrnsDict.keys() if k[0] == "Q"]:
                self.visualisationsPrnsDict[key].setChecked(True)
        else:
            for key in [k for k in self.visualisationsPrnsDict.keys() if k[0] == "Q"]:
                self.visualisationsPrnsDict[key].setChecked(False)
                
        if self.visualisationsChecksDict["I"].isChecked():
            for key in [k for k in self.visualisationsPrnsDict.keys() if k[0] == "I"]:
                self.visualisationsPrnsDict[key].setChecked(True)
        else:
            for key in [k for k in self.visualisationsPrnsDict.keys() if k[0] == "I"]:
                self.visualisationsPrnsDict[key].setChecked(False)
                
        if self.visualisationsChecksDict["C"].isChecked():
            for key in [k for k in self.visualisationsPrnsDict.keys() if k[0] == "C"]:
                self.visualisationsPrnsDict[key].setChecked(True)
        else:
            for key in [k for k in self.visualisationsPrnsDict.keys() if k[0] == "C"]:
                self.visualisationsPrnsDict[key].setChecked(False)
                
        if self.visualisationsChecksDict["S"].isChecked():
            for key in [k for k in self.visualisationsPrnsDict.keys() if k[0] == "S"]:
                self.visualisationsPrnsDict[key].setChecked(True)
        else:
            for key in [k for k in self.visualisationsPrnsDict.keys() if k[0] == "S"]:
                self.visualisationsPrnsDict[key].setChecked(False)
        
    def emptyTabUI(self):
        emptyTab = QWidget()
        layout = QVBoxLayout()

        emptyTab.setLayout(layout)
        return emptyTab
    
    def setGeneralInputs(self, parentLayout):
        flayout = QFormLayout()

        defaultStartDate = QDateTime(2023,2,23,0,0)
        defaultEndDate = QDateTime(2023,2,24,0,0)
        self.inputStartDate = QDateTimeEdit(defaultStartDate)
        flayout.addRow("Start Date", self.inputStartDate)

        self.inputEndDate = QDateTimeEdit(defaultEndDate)
        flayout.addRow("End Date", self.inputEndDate)

        hboxInterval = self.generateIntervalInputs()
        flayout.addRow("Interval", hboxInterval)

        hboxPosition = self.generateCoordinatesInputs()
        flayout.addRow("Observer's Position", hboxPosition)

        hboxMask = self.generateMaskInput()
        flayout.addRow("Mask", hboxMask)

        labelText = "Load Almanac File (.alm)"
        btn = QPushButton(labelText)
        btn.clicked.connect(self.loadAlmanac)
        flayout.addRow(btn)

        parentLayout.addLayout(flayout)

    def generateIntervalInputs(self):
        maxSeconds = 1_000_000
        maxMinutes = 1_000_000
        maxHours = 1_000_000
        maxDays = 1_000_000
        maxWeeks = 1_000_000
        minAmmountOfTime = 1

        secondValidator = QIntValidator(minAmmountOfTime, maxSeconds)
        minuteValidator = QIntValidator(minAmmountOfTime, maxMinutes)
        hourValidator = QIntValidator(minAmmountOfTime, maxHours)
        dayValidator = QIntValidator(minAmmountOfTime, maxDays)
        weekValidator = QIntValidator(minAmmountOfTime, maxWeeks)

        self.inputIntervalSeconds = QLineEdit()
        self.inputIntervalMinutes = QLineEdit()
        self.inputIntervalHours = QLineEdit("1")
        self.inputIntervalDays = QLineEdit()
        self.inputIntervalWeeks = QLineEdit()

        self.inputIntervalSeconds.setValidator(secondValidator)
        self.inputIntervalMinutes.setValidator(minuteValidator)
        self.inputIntervalHours.setValidator(hourValidator)
        self.inputIntervalDays.setValidator(dayValidator)
        self.inputIntervalWeeks.setValidator(weekValidator)

        self.inputIntervalSeconds.setPlaceholderText("Seconds")
        self.inputIntervalMinutes.setPlaceholderText("Minutes")
        self.inputIntervalHours.setPlaceholderText("Hours")
        self.inputIntervalDays.setPlaceholderText("Days")
        self.inputIntervalWeeks.setPlaceholderText("Weeks")

        hbox = QHBoxLayout()
        hbox.addWidget(self.inputIntervalSeconds)
        hbox.addWidget(self.inputIntervalMinutes)
        hbox.addWidget(self.inputIntervalHours)
        hbox.addWidget(self.inputIntervalDays)
        hbox.addWidget(self.inputIntervalWeeks)

        return hbox

    def generateCoordinatesInputs(self):
        minHeight = -100
        maxHeight = 1_000_000

        self.inputPhi = QLineEdit("21")
        self.inputLam = QLineEdit("52")
        self.inputHeight = QLineEdit("100")

        self.phiUnitCombobox = QComboBox()
        self.phiUnitCombobox.addItems(self.angleUnits)
        self.lamUnitCombobox = QComboBox()
        self.lamUnitCombobox.addItems(self.angleUnits)
        self.heightUnitCombobox = QComboBox()
        self.heightUnitCombobox.addItems(self.lenghtUnits)

        self.setPositionValidators()
        self.inputHeight.setValidator(QDoubleValidator(minHeight, maxHeight, 3, notation=QDoubleValidator.Notation.StandardNotation))

        self.inputPhi.setPlaceholderText("Latitude")
        self.inputLam.setPlaceholderText("Longitude")
        self.inputHeight.setPlaceholderText("Height")

        flayoutPosition = QFormLayout()

        phibox = QHBoxLayout()
        phibox.addWidget(self.inputPhi)
        phibox.addWidget(self.phiUnitCombobox)
        flayoutPosition.addRow("Latitude", phibox)

        lambox = QHBoxLayout()
        lambox.addWidget(self.inputLam)
        lambox.addWidget(self.lamUnitCombobox)
        flayoutPosition.addRow("Longitude", lambox)

        heightbox = QHBoxLayout()
        heightbox.addWidget(self.inputHeight)
        heightbox.addWidget(self.heightUnitCombobox)
        flayoutPosition.addRow("Height", heightbox)

        return flayoutPosition
    
    def setPositionValidators(self):
        maxPhiDeg = 90.0
        maxPhiRad = math.pi / 2
        maxPhiGrad = 100.0
        maxLamDeg = 180.0
        maxLamRad = math.pi
        maxLamGrad = 200.0

        phiDegValidator = QDoubleValidator(-maxPhiDeg, maxPhiDeg, 3, notation=QDoubleValidator.Notation.StandardNotation)
        phiRadValidator = QDoubleValidator(-maxPhiRad, maxPhiRad, 3, notation=QDoubleValidator.Notation.StandardNotation)
        phiGradValidator = QDoubleValidator(-maxPhiGrad, maxPhiGrad, 3, notation=QDoubleValidator.Notation.StandardNotation)
        lamDegValidator = QDoubleValidator(-maxLamDeg, maxLamDeg, 3, notation=QDoubleValidator.Notation.StandardNotation)
        lamRadValidator = QDoubleValidator(-maxLamRad, maxLamRad, 3, notation=QDoubleValidator.Notation.StandardNotation)
        lamGradValidator = QDoubleValidator(-maxLamGrad, maxLamGrad, 3, notation=QDoubleValidator.Notation.StandardNotation)

        if (self.phiUnitCombobox.currentText() == self.angleUnits[0]):
            self.inputPhi.setValidator(phiDegValidator)
        elif(self.phiUnitCombobox.currentText() == self.angleUnits[1]):
            self.inputPhi.setValidator(phiRadValidator)
        elif(self.phiUnitCombobox.currentText() == self.angleUnits[2]):
            self.inputPhi.setValidator(phiGradValidator)
        else:
            raise ValueError("Wrong angle units")
        
        if (self.lamUnitCombobox.currentText() == self.angleUnits[0]):
            self.inputLam.setValidator(lamDegValidator)
        elif(self.lamUnitCombobox.currentText() == self.angleUnits[1]):
            self.inputLam.setValidator(lamRadValidator)
        elif(self.lamUnitCombobox.currentText() == self.angleUnits[2]):
            self.inputLam.setValidator(lamGradValidator)
        else:
            raise ValueError("Wrong angle units")

    def getPosition(self):
        if ("" in [self.inputPhi.text(), self.inputLam.text(), self.inputHeight.text()]):
            return []
        
        if (self.phiUnitCombobox.currentText() == self.angleUnits[0]):
            phi = self.textEdit2Float(self.inputPhi)
        elif(self.phiUnitCombobox.currentText() == self.angleUnits[1]):
            phi = np.rad2deg(self.textEdit2Float(self.inputPhi))
        elif(self.phiUnitCombobox.currentText() == self.angleUnits[2]):
            phi = self.textEdit2Float(self.inputPhi) / 200 * 180
        else:
            raise ValueError("Wrong angle units")
        
        if (self.lamUnitCombobox.currentText() == self.angleUnits[0]):
            lam = self.textEdit2Float(self.inputLam)
        elif(self.lamUnitCombobox.currentText() == self.angleUnits[1]):
            lam = np.rad2deg(self.textEdit2Float(self.inputLam))
        elif(self.lamUnitCombobox.currentText() == self.angleUnits[2]):
            lam = self.textEdit2Float(self.inputLam) / 200 * 180
        else:
            raise ValueError("Wrong angle units")
        
        feet2meterRatio = 0.3048

        if (self.heightUnitCombobox.currentText() == self.lenghtUnits[0]):
            height = self.textEdit2Float(self.inputHeight)
        elif (self.heightUnitCombobox.currentText() == self.lenghtUnits[1]):
            height = self.textEdit2Float(self.inputHeight) * feet2meterRatio

        return [phi, lam, height]
    
    def generateMaskInput(self):
        minMask = 0
        maxMaskDeg = 90
        maxMaskRad = math.pi / 2
        maxMaskGrad = 100 

        self.inputMask = QLineEdit("10")

        self.maskUnitCombobox = QComboBox()
        self.maskUnitCombobox.addItems(self.angleUnits)

        if (self.maskUnitCombobox.currentText() == self.angleUnits[0]):
            self.inputMask.setValidator(QDoubleValidator(minMask, maxMaskDeg, 3, notation=QDoubleValidator.Notation.StandardNotation))
        elif(self.maskUnitCombobox.currentText() == self.angleUnits[1]):
            self.inputMask.setValidator(QDoubleValidator(minMask, maxMaskRad, 3, notation=QDoubleValidator.Notation.StandardNotation))
        elif(self.maskUnitCombobox.currentText() == self.angleUnits[2]):
            self.inputMask.setValidator(QDoubleValidator(minMask, maxMaskGrad, 3, notation=QDoubleValidator.Notation.StandardNotation))
        else:
            raise ValueError("Wrong angle units")
        
        self.inputMask.setPlaceholderText("Mask")

        maskbox = QHBoxLayout()
        maskbox.addWidget(self.inputMask)
        maskbox.addWidget(self.maskUnitCombobox)

        return maskbox  

    def getMask(self):
        if ("" == self.inputMask.text()):
            return -1
        
        if (self.maskUnitCombobox.currentText() == self.angleUnits[0]):
            mask = self.textEdit2Float(self.inputMask)
        elif(self.maskUnitCombobox.currentText() == self.angleUnits[1]):
            mask = np.rad2deg(self.textEdit2Float(self.inputMask))
        elif(self.maskUnitCombobox.currentText() == self.angleUnits[2]):
            mask = self.textEdit2Float(self.inputMask) / 200 * 180
        else:
            raise ValueError("Wrong angle units")
    
        return mask

    def getStartDate(self):
       pyqtdate = self.inputStartDate.dateTime().toString()
       out = self.string2datetime(pyqtdate)
       return out

    def getEndDate(self):   
       pyqtdate = self.inputEndDate.dateTime().toString()
       out = self.string2datetime(pyqtdate)
       return out

    def string2datetime(self, string : 'str'):
        list = string.split(' ')
        month = list[1]
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Oct", "Sep", "Nov", "Dec"]
        for i, m in enumerate(months):
            if month == m:
                month = i + 1
                break
        
        time = list[3].split(":")

        return datetime(int(list[4]), int(month), int(list[2]), int(time[0]), int(time[1]), int(time[2]))
        
    def getInterval(self):
        if ("" == self.inputIntervalSeconds.text() 
        and "" ==  self.inputIntervalMinutes.text() 
        and "" ==  self.inputIntervalHours.text() 
        and "" ==  self.inputIntervalDays.text() 
        and "" ==   self.inputIntervalWeeks.text()):
            return timedelta(0)

        sec, min, hour, day, week = 0, 0, 0, 0, 0

        if "" != self.inputIntervalSeconds.text():
            sec = int(self.inputIntervalSeconds.text())
        if "" != self.inputIntervalMinutes.text():
            min = int(self.inputIntervalMinutes.text())
        if "" != self.inputIntervalHours.text():
            hour = int(self.inputIntervalHours.text())
        if "" != self.inputIntervalDays.text():
            day = int(self.inputIntervalDays.text())
        if "" != self.inputIntervalWeeks.text():
            week = int(self.inputIntervalWeeks.text())

        return timedelta(days=(week*7 + day), hours=hour, minutes=min, seconds=sec)
    
    def checkInputs(self):
        startDate = self.getStartDate()
        endDate = self.getEndDate()
        interval = self.getInterval()
        position = self.getPosition()
        mask = self.getMask()

        datesOk = startDate <= endDate
        intervalOk = interval > timedelta(0) and startDate + interval < endDate
        positionOk = bool(position)
        maskOk = mask >= 0
        fileNotEmpty = self.almanacFilePath != ""

        return [datesOk, intervalOk, positionOk, maskOk, fileNotEmpty]

    def updateInputsStateLabel(self):
        fileNotEmpty = self.almanacFilePath != ""

        inputsStates = self.checkInputs()
        everythingOk = all(inputsStates)

        newText = f"Start date before End Date: {(inputsStates[0])} || Interval: {inputsStates[1]} || Position: {inputsStates[2]}  || Mask: {inputsStates[3]} || Almanac file: {inputsStates[4]}"

        self.labelInputState.setText(newText)

        return everythingOk

    def calculate(self):
        if not all(self.checkInputs()):
            self.promptUserToFixInputs()
            return
         
        startDate = self.getStartDate()
        endDate = self.getEndDate()
        mask = self.getMask()
        position = self.getPosition()
        interval = self.getInterval()
        try:
            self.prns, self.satellitesPositionsForWholePeriod, self.satellitesVisibleOverWholePeriod, self.matricesAAndDOPsOverWholePerid = createSatellitesVisibleSatellitesPositions(
                startDate, 
                endDate, 
                interval, 
                position[0], 
                position[1], 
                position[2], 
                mask, self.almanacFilePath)
            self.mask = mask

            self.updatePrnsCheckboxesInInputTab()
            self.promptCalculated()

        except:
            self.promptNotCalculated()

    def checkCalculated(self):
        return [len(self.prns)!=0, len(self.satellitesPositionsForWholePeriod)!=0, len(self.satellitesVisibleOverWholePeriod)!=0, len(self.matricesAAndDOPsOverWholePerid)!=0]

    def clearLayout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def clear(self):
        self.clearLayout(self.visualisationsGnssChoiceLayout)
        self.clearLayout(self.visualisationsPrnsLayout)

        
        self.inputIntervalDays.clear()
        self.inputIntervalHours.clear()
        self.inputIntervalMinutes.clear()
        self.inputIntervalSeconds.clear()
        self.inputIntervalWeeks.clear()
        self.inputLam.clear()
        self.inputPhi.clear()
        self.inputHeight.clear()
        self.inputMask.clear()

        self.setVariables()

        self.updateInputsStateLabel()

    def loadAlmanac(self):
        self.almanacFilePath = self.load()

### Visualisations
    def getPrnsToVisualise(self, prnsDict : 'dict'):
        prnsToVisualise = []
        for key in prnsDict.keys():
            if prnsDict[key].isChecked():
                prnsToVisualise.append(key)
        return prnsToVisualise

    def getSatellitesToVisualise(self, satellitesPositionsForWholePeriod, visualisationsPrnsDict):
        visualisationTable = []
        for epoch in satellitesPositionsForWholePeriod:
            visualisationTableForOneEpoch = []
            for i, prn in enumerate(self.prns):
                if not visualisationsPrnsDict[prn].isChecked():
                    continue
                visualisationTableForOneEpoch.append(epoch[i])
            visualisationTable.append(visualisationTableForOneEpoch)
        return visualisationTable

    def visualiseSphere(self):
        if all(self.checkCalculated()):
            visualisationTable = self.getSatellitesToVisualise(self.satellitesPositionsForWholePeriod, self.visualisationsPrnsDict)
            try:
                plot3dSatellitesOnAxis(self.sphereAx, np.array(visualisationTable), self.prns, self.sphereLegendCheckBox.isChecked())
                self.sphereCanvas.draw()
            except:
                self.promptError()
        else:
            self.promptUserToCalculate()

    def sphereVisualisationTabUI(self):
        sphereVisualisationTab = QWidget()
        layout = QVBoxLayout()

        hlayout = QHBoxLayout()

        self.sphereLegendCheckBox = QCheckBox("Legend")
        hlayout.addWidget(self.sphereLegendCheckBox)
        
        tmpbtn = QPushButton("Update visualisation")
        tmpbtn.clicked.connect(self.visualiseSphere)
        hlayout.addWidget(tmpbtn)

        layout.addLayout(hlayout)

        visualisationLayout = QVBoxLayout()
        fig = Figure(figsize=(5, 5))
        fig.tight_layout()
        self.sphereCanvas = FigureCanvas(fig)
        self.sphereToolbar = NavigationToolbar(self.sphereCanvas, self)
        fig.set_canvas(self.sphereCanvas)

        self.sphereAx = self.sphereCanvas.figure.add_subplot(projection="3d")
        self.sphereAx.set_xlabel("X axis")
        self.sphereAx.set_ylabel("Y axis")
        self.sphereAx.set_zlabel("Z axis")
        
        visualisationLayout.addWidget(self.sphereToolbar)
        visualisationLayout.addWidget(self.sphereCanvas)

        layout.addLayout(visualisationLayout)

        sphereVisualisationTab.setLayout(layout)
        return sphereVisualisationTab
        
    def visualiseSkyPlot(self):
        if all(self.checkCalculated()):
            if self.skyPlotNoEpochsInput.text() == "": numberOfEpochs = 1
            else: numberOfEpochs = int(self.skyPlotNoEpochsInput.text())
            if numberOfEpochs == 0: numberOfEpochs = 1
            
            if self.skyPlotFirstEpochNumber.text() == "": epochNo = 0
            else: epochNo = int(self.skyPlotFirstEpochNumber.text())
            if epochNo < 0: epochNo = 0

            if epochNo + numberOfEpochs > len(self.satellitesVisibleOverWholePeriod):
                self.promptWrongEpochsNumbers()
                return False

            visualisationTable = []
            for i in range(len(self.satellitesVisibleOverWholePeriod)):
                newRow = []
                for j in range(len(self.satellitesVisibleOverWholePeriod[i])):
                    elevation = self.satellitesVisibleOverWholePeriod[i][j][2]
                    if np.rad2deg(elevation) < self.mask:
                        continue
                    
                    prn = self.satellitesVisibleOverWholePeriod[i][j][1]
                    if not self.visualisationsPrnsDict[prn].isChecked():
                        continue
                    newRow.append(self.satellitesVisibleOverWholePeriod[i][j])
                visualisationTable.append(newRow)

            plotSkyplotFromSatVisibleOnAxis(self.skyPlotAx, visualisationTable, startEpochNo=epochNo ,numberOfEpochs=numberOfEpochs, fontsize=10)
            self.skyPlotCanvas.draw()
        else:
            self.promptUserToCalculate()
    
    def skyPlotVisualisationTabUI(self):
        skyPlotVisualisationTab = QWidget()
        layout = QVBoxLayout()

        hlayout = QHBoxLayout()
        tmpbtn = QPushButton("Update visualisation")
        tmpbtn.clicked.connect(self.visualiseSkyPlot)
        hlayout.addWidget(tmpbtn)

        self.skyPlotFirstEpochNumber = QLineEdit()
        self.skyPlotFirstEpochNumber.setPlaceholderText("First epoch number")
        skyPlotFirstEpochNumberValidator = QIntValidator()
        skyPlotFirstEpochNumberValidator.setBottom(0)
        self.skyPlotFirstEpochNumber.setValidator(skyPlotFirstEpochNumberValidator)

        hlayout.addWidget(self.skyPlotFirstEpochNumber)

        self.skyPlotNoEpochsInput = QLineEdit()
        self.skyPlotNoEpochsInput.setPlaceholderText("Number of epochs")
        skyPlotnoEpochsValidator = QIntValidator()
        skyPlotnoEpochsValidator.setBottom(1)
        self.skyPlotNoEpochsInput.setValidator(skyPlotnoEpochsValidator)

        hlayout.addWidget(self.skyPlotNoEpochsInput)

        layout.addLayout(hlayout)

        visualisationLayout = QVBoxLayout()
        fig = Figure(figsize=(10, 6))
        self.skyPlotCanvas = FigureCanvas(fig)
        fig.set_canvas(self.skyPlotCanvas)
        fig.tight_layout()
        self.skyPlotAx = self.skyPlotCanvas.figure.add_subplot(polar=True)
        
        visualisationLayout.addWidget(self.skyPlotCanvas)
        layout.addLayout(visualisationLayout)

        skyPlotVisualisationTab.setLayout(layout)
        return skyPlotVisualisationTab

    def visualiseGroundtrack(self):
        if all(self.checkCalculated()):
            if self.groundtrackNoEpochsInput.text() == "": numberOfEpochs = 1
            else: numberOfEpochs = int(self.groundtrackNoEpochsInput.text())
            if numberOfEpochs == 0: numberOfEpochs = 1
            
            if self.groundtrackFirstEpochNumber.text() == "": epochNo = 0
            else: epochNo = int(self.groundtrackFirstEpochNumber.text())
            if epochNo < 0: epochNo = 0

            if epochNo + numberOfEpochs > len(self.satellitesPositionsForWholePeriod):
                self.promptWrongEpochsNumbers()
                return False

            visualisationTable = self.getSatellitesToVisualise(self.satellitesPositionsForWholePeriod, self.visualisationsPrnsDict)
            visualisationTable = visualisationTable[epochNo:epochNo + numberOfEpochs]
            satellitesPhiLam = satellitesPositionsForWholePeriod2satellitesPhiLam(visualisationTable)
            prns = self.getPrnsToVisualise(self.visualisationsPrnsDict)

            coastlineDataPath = pathlib.Path(__file__).parent.resolve()/'Coastline.txt'
            self.groundtrackAx.cla()
            for i, satellite in enumerate(satellitesPhiLam):
                plotGroundtrackOnAxis(prns[i], satellite, self.groundtrackAx, coastlineDataPath, 10)
            self.groundtrackCanvas.draw()
        else:
            self.promptUserToCalculate()

    def groundtrackTabUI(self):
        groundtrackTab = QWidget()
        layout = QVBoxLayout()

        hlayout = QHBoxLayout()

        tmpbtn = QPushButton("Update visualisation")
        tmpbtn.clicked.connect(self.visualiseGroundtrack)
        hlayout.addWidget(tmpbtn)

        self.groundtrackFirstEpochNumber = QLineEdit()
        self.groundtrackFirstEpochNumber.setPlaceholderText("First epoch number")
        groundtrackEpochNumberValidator = QIntValidator()
        groundtrackEpochNumberValidator.setBottom(0)
        self.groundtrackFirstEpochNumber.setValidator(groundtrackEpochNumberValidator)

        hlayout.addWidget(self.groundtrackFirstEpochNumber)

        self.groundtrackNoEpochsInput = QLineEdit()
        self.groundtrackNoEpochsInput.setPlaceholderText("Number of epochs")
        groundtrackNoEpochsValidator = QIntValidator()
        groundtrackNoEpochsValidator.setBottom(1)
        self.groundtrackNoEpochsInput.setValidator(groundtrackNoEpochsValidator)

        hlayout.addWidget(self.groundtrackNoEpochsInput)

        layout.addLayout(hlayout)

        visualisationLayout = QVBoxLayout()
        fig = Figure(figsize=(8,4))
        fig.tight_layout()
        self.groundtrackCanvas = FigureCanvas(fig)
        self.groundtrackToolbar = NavigationToolbar(self.groundtrackCanvas, self)
        fig.set_canvas(self.groundtrackCanvas)

        self.groundtrackAx = self.groundtrackCanvas.figure.add_subplot(1,1,1)
        
        self.groundtrackAx.set_ylim(-90,90);
        self.groundtrackAx.set_xlim(-180,180);
        self.groundtrackAx.set_yticks([-90,-80,-70,-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60,70,80,90]);
        self.groundtrackAx.set_xticks([-180,-150,-120,-90,-60,-30,0,30,60,90,120,150,180]);

        visualisationLayout.addWidget(self.groundtrackToolbar)
        visualisationLayout.addWidget(self.groundtrackCanvas)    

        layout.addLayout(visualisationLayout)

        groundtrackTab.setLayout(layout)
        return groundtrackTab

    def visualiseNumberOfVisibleSatellites(self):
        if all(self.checkCalculated()):
            visualisationTable = []
            for epoch in self.satellitesVisibleOverWholePeriod:
                visualisationTableForOneEpoch = []
                for satellite in epoch:
                    elevation = satellite[2]
                    if np.rad2deg(elevation) < self.mask:
                        continue

                    prn = satellite[1]
                    if not self.visualisationsPrnsDict[prn].isChecked():
                        continue
                    visualisationTableForOneEpoch.append(satellite)
                visualisationTable.append(visualisationTableForOneEpoch)

            plotVisibleSatellitesNumberOnAxis(self.noOfVisibleSatellitesAx, visualisationTable)
            self.noOfVisibleSatellitesCanvas.draw()
        else:
            self.promptUserToCalculate()

    def numberOfVisibleSatellitesVisualisationTabUI(self):
        numberOfVisibleSatellitesTab = QWidget()
        layout = QVBoxLayout()

        hlayout = QHBoxLayout()
        tmpbtn = QPushButton("Update visualisation")
        tmpbtn.clicked.connect(self.visualiseNumberOfVisibleSatellites)
        hlayout.addWidget(tmpbtn)

        layout.addLayout(hlayout)

        visualisationLayout = QVBoxLayout()
        fig = Figure(figsize=(8,4))
        fig.tight_layout()
        self.noOfVisibleSatellitesCanvas = FigureCanvas(fig)
        self.noOfVisibleSatellitesToolbar = NavigationToolbar(self.groundtrackCanvas, self)
        fig.set_canvas(self.noOfVisibleSatellitesCanvas)

        self.noOfVisibleSatellitesAx = self.noOfVisibleSatellitesCanvas.figure.add_subplot(1,1,1)
        self.noOfVisibleSatellitesAx.grid()


        visualisationLayout.addWidget(self.noOfVisibleSatellitesToolbar)
        visualisationLayout.addWidget(self.noOfVisibleSatellitesCanvas)    

        layout.addLayout(visualisationLayout)

        numberOfVisibleSatellitesTab.setLayout(layout)
        return numberOfVisibleSatellitesTab
    
    def visualiseDOPs(self):
        if all(self.checkCalculated()):
            plotDOPsOnAxis(self.DOPsAx, [x[1] for x in self.matricesAAndDOPsOverWholePerid])
            self.DOPsCanvas.draw()
        else:
            self.promptUserToCalculate()

    def DOPsTabUI(self):
        DOPsTab = QWidget()
        layout = QVBoxLayout()

        hlayout = QHBoxLayout()
        tmpbtn = QPushButton("Update visualisation")
        tmpbtn.clicked.connect(self.visualiseDOPs)
        hlayout.addWidget(tmpbtn)

        layout.addLayout(hlayout)

        visualisationLayout = QVBoxLayout()
        fig = Figure(figsize=(8,4))
        fig.tight_layout()
        self.DOPsCanvas = FigureCanvas(fig)
        self.DOPsToolbar = NavigationToolbar(self.DOPsCanvas, self)
        fig.set_canvas(self.DOPsCanvas)

        self.DOPsAx = self.DOPsCanvas.figure.add_subplot(1,1,1)

        visualisationLayout.addWidget(self.DOPsToolbar)
        visualisationLayout.addWidget(self.DOPsCanvas)    

        layout.addLayout(visualisationLayout)

        DOPsTab.setLayout(layout)
        return DOPsTab
    
    def visualiseElevations(self):
        if all(self.checkCalculated()):
            if self.elevationsNoEpochsInput.text() == "": numberOfEpochs = len(self.satellitesPositionsForWholePeriod)
            else: numberOfEpochs = int(self.elevationsNoEpochsInput.text())
            if numberOfEpochs == 0: numberOfEpochs = 1
            
            if self.elevationsFirstEpochNumber.text() == "": epochNo = 0
            else: epochNo = int(self.elevationsFirstEpochNumber.text())
            if epochNo < 0: epochNo = 0

            visualisationTable = []
            for i, epoch in enumerate(self.satellitesVisibleOverWholePeriod):
                if i < epochNo: continue
                elif i >= epochNo + numberOfEpochs: continue
                visualisationTableForOneEpoch = []
                for satellite in epoch:
                    prn = satellite[1]
                    if not self.visualisationsPrnsDict[prn].isChecked():
                        continue
                    visualisationTableForOneEpoch.append(satellite)
                visualisationTable.append(visualisationTableForOneEpoch)


            plotElevationsOnAxis(self.elevationsAx, visualisationTable, legend=self.elevationsLegendCheckBox.isChecked())
            self.elevationsCanvas.draw()
        else:
            self.promptUserToCalculate()

    def elevationsTabUI(self):
        DOPsTab = QWidget()
        layout = QVBoxLayout()

        hlayout = QHBoxLayout()
        tmpbtn = QPushButton("Update visualisation")
        tmpbtn.clicked.connect(self.visualiseElevations)
        hlayout.addWidget(tmpbtn)
        
        self.elevationsLegendCheckBox = QCheckBox("Legend")
        hlayout.addWidget(self.elevationsLegendCheckBox)

        self.elevationsFirstEpochNumber = QLineEdit()
        self.elevationsFirstEpochNumber.setPlaceholderText("First epoch number")
        epevationsEpochNumberValidator = QIntValidator()
        epevationsEpochNumberValidator.setBottom(0)
        self.elevationsFirstEpochNumber.setValidator(epevationsEpochNumberValidator)
        hlayout.addWidget(self.elevationsFirstEpochNumber)

        self.elevationsNoEpochsInput = QLineEdit()
        self.elevationsNoEpochsInput.setPlaceholderText("Number of epochs")
        elevationsEpochsValidator = QIntValidator()
        elevationsEpochsValidator.setBottom(1)
        self.elevationsNoEpochsInput.setValidator(elevationsEpochsValidator)
        hlayout.addWidget(self.elevationsNoEpochsInput)

        layout.addLayout(hlayout)

        visualisationLayout = QVBoxLayout()
        fig = Figure(figsize=(8,4))
        fig.tight_layout()
        self.elevationsCanvas = FigureCanvas(fig)
        self.elevationsToolbar = NavigationToolbar(self.elevationsCanvas, self)
        fig.set_canvas(self.elevationsCanvas)

        self.elevationsAx = self.elevationsCanvas.figure.add_subplot(1,1,1)

        visualisationLayout.addWidget(self.elevationsToolbar)
        visualisationLayout.addWidget(self.elevationsCanvas)    

        layout.addLayout(visualisationLayout)

        DOPsTab.setLayout(layout)
        return DOPsTab


### Universals
    def textEdit2Float(self, textEdit):
        return float(textEdit.text().replace(',', '.'))

    def save(self, data):
        if len(data) == 0:
            self.promptError("Nothing to save.")
            return False
        
        fname = self.new_file_dialog()
        if fname[1] == "":
            return False
            
        if fname[1] != "Almanac Files (*.alm)":
            self.promptError("Wrong file. Chceck if it has correct extension.")
            return False

        if not fname[0].endswith(".alm"):
            fname = list(fname)
            fname[0] += ".alm"
        
        with open(fname[0], "w") as outfile:
            data.to_csv(path_or_buf=outfile,sep=";")

    def load(self):
        fname = self.open_file_dialog()
        if fname[1] not in ["Almanac Files (*.alm)", ""]:
            self.displayError()
            return ""
        elif fname[1] == "":
            return ""
        else:
            return fname[0]

    def open_file_dialog(self):
        fname = QFileDialog.getOpenFileName(
            self,
            "Open File",
            os.path.dirname(__file__),
            "Almanac Files (*.alm)"
        )
        return fname

    def new_file_dialog(self):
        fname = QFileDialog.getSaveFileName(
            self,
            "Save as",
            os.path.dirname(__file__),
            "Almanac Files (*.alm)"
        )
        return fname

### Prompts
    def promptError(self, msg=None):
        dlg = QMessageBox(self)
        dlg.setIcon(QMessageBox.Icon.Critical)
        dlg.setWindowTitle("Error")
        if msg:
            dlg.setText(msg)
        else:
            dlg.setText("Application encountered an error. If you've done something you probably shouldn't have, please don't do that again.\n\n" +
                    "'Insanity: doing the same thing over and over again and expecting different results.'\n                 -Albert Einstein")
        dlg.exec()
        self.tabsWid.setCurrentIndex(0)

    def promptUserToFixInputs(self):
        self.tabsWid.setCurrentIndex(0)
        self.promptCustom("Please make sure all the fields are filled correctly. \nYou may want to use 'check inputs' button.", 
                          "Fix inputs", QMessageBox.Icon.Warning)

    def promptUserToCalculate(self):
        self.promptCustom("Please use 'calculate' button before visualising.", "Error", QMessageBox.Icon.Critical)

    def promptCalculated(self):
        self.promptCustom("Calculated successfully.", "Calculated", QMessageBox.Icon.Information)
        self.tabsWid.setCurrentIndex(0)

    def promptNotCalculated(self):
        dlg = QMessageBox(self)
        dlg.setIcon(QMessageBox.Icon.Critical)
        dlg.setWindowTitle("Prompt")
        dlg.setText("Calculation was unsuccessfull.")
        dlg.exec()
        self.tabsWid.setCurrentIndex(0)

    def promptWrongEpochsNumbers(self):
        self.promptCustom(("You've entered wrong number of epochs.\n" + 
                           f"Note that you have {len(self.satellitesPositionsForWholePeriod)} calculated right now and automatically first epoch's number is 0."),
                           "Wrong number of epochs", QMessageBox.Icon.Critical)

    def promptCustom(self, message : 'str', title : 'str' = "Prompt", icon : 'QMessageBox.Icon' = QMessageBox.Icon.Information):
        dlg = QMessageBox(self)
        dlg.setIcon(icon)
        dlg.setWindowTitle(title)
        dlg.setText(message)
        dlg.exec()
        self.tabsWid.setCurrentIndex(0)

app = QApplication(sys.argv)
app.setStyleSheet(
    '''
    QWidget{
        font-size: 15px;
    }
    QLabel{
        text-align: center;
    }
    '''
)
window = MyApp()
window.show()

app.exec()