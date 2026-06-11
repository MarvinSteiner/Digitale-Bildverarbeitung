import cv2
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt

from . import Algorithm


class InvisCloak (Algorithm):

    """ init function """
    def __init__(self):
        self.image_stack = list()
        self.background = None
        self.capture_background = False

    """ Processes the input image"""
    def process(self, img):

        """ 2.1 Vorverarbeitung """
        """ 2.1.1 Rauschreduktion """
        plotNoise = False   # Schaltet die Rauschvisualisierung ein
        if plotNoise:
            self._plotNoise(img, "Rauschen vor Korrektur")
        img = self._211_Rauschreduktion(img)
        if plotNoise:
            self._plotNoise(img, "Rauschen nach Korrektur")
        """ 2.1.2 HistogrammSpreizung """
        img = self._212_HistogrammSpreizung(img)


        """ 2.2 Farbanalyse """
        """ 2.2.1 RGB """
        self._221_RGB(img)
        """ 2.2.2 HSV """
        self._222_HSV(img)


        """ 2.3 Segmentierung und Bildmdifikation """
        img = self._23_SegmentUndBildmodifizierung(img)

        return img

    """ Reacts on mouse callbacks """
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            print("A Mouse click happend! at position", x, y)
            self.capture_background = True

    def _plotNoise(self, img, name:str):
        height, width = np.array(img.shape[:2])
        centY = (height / 2).astype(int)
        centX = (width / 2).astype(int)

        cutOut = 5
        tmpImg = deepcopy(img)
        tmpImg = tmpImg[centY - cutOut:centY + cutOut, centX - cutOut:centX + cutOut, :]

        outSize = 500
        tmpImg = cv2.resize(tmpImg, (outSize, outSize), interpolation=cv2.INTER_NEAREST)

        cv2.imshow(name, tmpImg)
        cv2.waitKey(1)

    def _211_Rauschreduktion(self, img):
        """
            Hier steht Ihr Code zu Aufgabe 2.1.1 (Rauschunterdrückung)
            - Implementierung Mittelwertbildung über N Frames
        """

        N = 1  # Default laut Aufgabenstellung 2.1.1

        # Füge Bilder zum Image Stack hinzu, sodass maximal max_image_stack_length Bilder dort gespeichert sind
        max_image_stack_length = N + 1
        self.image_stack.append(img)
        if len(self.image_stack) > max_image_stack_length:
            self.image_stack = self.image_stack[-max_image_stack_length:]

        # Mittelwert über alle Bilder im Stack berechnen
        stack = np.stack(self.image_stack, axis=0)

        # Überläufe durch 32-Bit Int vermeiden
        sum_img = np.sum(stack.astype(np.uint32), axis=0)

        # Anzahl der aktuell gespeicherten Bilder
        num_images = len(self.image_stack)

        # Mittelwert
        mean_img = sum_img // num_images

        # Zurück in 8-Bit konvertieren
        img = mean_img.astype(np.uint8)

        return img

    def _212_HistogrammSpreizung(self, img):
        """
            Hier steht Ihr Code zu Aufgabe 2.1.2 (Histogrammspreizung)
            - Transformation HSV
            - Histogrammspreizung berechnen
            - Transformation BGR
        """
        #von BGR in HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        #Histogrammspreizung nur von Value (Helligkeit)
        v = cv2.normalize(
            v,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX
        )

        #Zusammensetzung zu HSV nach Histogrammspreizung
        hsv = cv2.merge((h, s, v))

        #von HSV in BGR (weiterhin mit dem Kamera-Bild arbeiten)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def _221_RGB(self, img):
        """
            Hier steht Ihr Code zu Aufgabe 2.2.1 (RGB)
            - Histogrammberechnung und Analyse
        """
        #BGR
        b_hist = cv2.calcHist([img], [0], None, [256], [0, 256]) #blau Kanal 0
        g_hist = cv2.calcHist([img], [1], None, [256], [0, 256]) #grün Kanal 1
        r_hist = cv2.calcHist([img], [2], None, [256], [0, 256]) #rot Kanal 2

        return r_hist, g_hist, b_hist

    def _222_HSV(self, img):
        """
            Hier steht Ihr Code zu Aufgabe 2.2.2 (HSV)
            - Histogrammberechnung und Analyse im HSV-Raum
        """
        #von BGR in HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        #HSV
        h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180]) #Hue (Farbton) Kanal 0

        return h_hist

    def _23_SegmentUndBildmodifizierung(self, img):
        """
            Hier steht Ihr Code zu Aufgabe 2.3.1 (StatischesSchwellwertverfahren)
            - Binärmaske erstellen
        """
        #von BGR in HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        #Rot Range 1
        lower_red1 = np.array([0, 50, 40])
        upper_red1 = np.array([10, 255, 255])

        #Rot Range 2
        lower_red2 = np.array([170, 50, 40])
        upper_red2 = np.array([180, 255, 255])

        #Binärmaske erstellen
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2

        """
            Hier steht Ihr Code zu Aufgabe 2.3.2 (Binärmaske)
            - Binärmaske optimieren mit Opening/Closing
            - Wahl größte zusammenhängende Region
        """
        #5x5 Matrix Kernel (nur 1)
        kernel = np.ones((5, 5), np.uint8)

        #Opening (Erosion + Dilatation)
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel,
            iterations=2
        )

        #Closing (Dilatation + Erosion)
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2
        )

        #Außenlinie (Kontur) finden
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL, #nur äußere Konturen, keine inneren Konturen
            cv2.CHAIN_APPROX_SIMPLE
        )

        #Leere Maske erstellen für Speicherung der größten Region
        largest_mask = np.zeros_like(mask)

        if len(contours) > 0: #Bugfix, falls kein Umhang sichtbar ist
            #Fläche jeder Kontur berechnen
            largest_contour = max(contours, key=cv2.contourArea)

            #größte Kontur in der leeren Bitmaske einfärben
            cv2.drawContours(
                largest_mask,
                [largest_contour],
                -1,
                255,
                thickness=cv2.FILLED #Umhang ist ganze Fläche, nicht nur Außenlinie
            )

            #Annahme: Umhang ist größte rote Kontur im Bild (größtes Objekt flächenmäßig)
            mask = largest_mask

        """
            Hier steht Ihr Code zu Aufgabe 2.3.1 (Bildmodifizerung)
            - Hintergrund mit Mausklick definieren
            - Ersetzen des Hintergrundes
        """
        #Hintergrund aufnehmen
        if self.capture_background:
            self.background = img.copy() #copy sonst jedes neue Kamerabild = neuer Hintergrund
            self.capture_background = False

        #Falls noch kein Hintergrund aufgenommen wurde
        if self.background is None:
            return img

        #Maske invertieren
        mask_inv = cv2.bitwise_not(mask)

        #Vordergrund extrahieren (alles außer Umhang)
        foreground = cv2.bitwise_and(
            img,
            img,
            mask=mask_inv
        )

        #Hintergrund extrahieren aus Momentaufnahme beim Mausklick (Bereich des Umhangs)
        background_part = cv2.bitwise_and(
            self.background,
            self.background,
            mask=mask
        )

        #Vorder- und  Hintergrund zusammenfügen
        result = cv2.add(
            foreground,
            background_part
        )

        return result