import sys
import json
import math
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QWidget, QPushButton, QLabel, QFileDialog, QListWidget,
                             QListWidgetItem, QColorDialog, QInputDialog, QMessageBox,
                             QScrollArea, QFrame)
from PyQt5.QtCore import Qt, QRect, QPoint, pyqtSignal
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor, QPixmap, QFont, QPolygon, QWheelEvent
import os


class ZoomableAnnotationWidget(QLabel):
    """Custom widget for drawing and editing polygons on images with zoom and pan"""

    def __init__(self):
        super().__init__()
        self.panning_image = False  # Add this line
        self.original_image = None
        self.scaled_image = None
        self.zoom_factor = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 10.0

        # Image positioning
        self.image_offset = QPoint(0, 0)  # Offset of image within widget

        self.polygons = []  # List of polygons, each containing points in IMAGE coordinates
        self.selected_polygon = -1  # Index of selected polygon
        self.selected_vertex = -1  # Index of selected vertex
        self.vertex_radius = 5

        # Pan and interaction state
        self.pan_start = None
        self.dragging_vertex = False
        self.dragging_polygon = False
        self.drag_start_pos = None
        self.polygon_drag_offset = QPoint(0, 0)

        self.setMinimumSize(800, 600)
        self.setStyleSheet("border: 1px solid gray;")
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)

    def load_image(self, image_path):
        """Load an image for annotation"""
        self.original_image = QPixmap(image_path)
        if not self.original_image.isNull():
            self.zoom_factor = 1.0
            self.update_scaled_image()
            self.polygons = []
            self.selected_polygon = -1
            self.selected_vertex = -1
            self.update()

    def update_scaled_image(self):
        """Update the scaled image based on current zoom factor"""
        if self.original_image:
            new_size = self.original_image.size() * self.zoom_factor
            self.scaled_image = self.original_image.scaled(
                new_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setPixmap(self.scaled_image)
            self.resize(self.scaled_image.size())

            # Calculate image offset for centering
            widget_rect = self.rect()
            image_rect = self.scaled_image.rect()
            self.image_offset = QPoint(
                (widget_rect.width() - image_rect.width()) // 2,
                (widget_rect.height() - image_rect.height()) // 2
            )

    def canvas_to_image_coords(self, canvas_point):
        """Convert canvas coordinates to image coordinates"""
        if not self.original_image:
            return canvas_point

        # Adjust for image offset and zoom
        image_x = (canvas_point.x() - self.image_offset.x()) / self.zoom_factor
        image_y = (canvas_point.y() - self.image_offset.y()) / self.zoom_factor

        # Clamp to image bounds
        image_x = max(0, min(self.original_image.width() - 1, image_x))
        image_y = max(0, min(self.original_image.height() - 1, image_y))

        return QPoint(int(image_x), int(image_y))

    def image_to_canvas_coords(self, image_point):
        """Convert image coordinates to canvas coordinates"""
        if not self.original_image:
            return image_point

        canvas_x = image_point.x() * self.zoom_factor + self.image_offset.x()
        canvas_y = image_point.y() * self.zoom_factor + self.image_offset.y()

        return QPoint(int(canvas_x), int(canvas_y))

    def get_image_rect(self):
        """Get the rectangle of the image in canvas coordinates"""
        if not self.scaled_image:
            return QRect()

        return QRect(self.image_offset, self.scaled_image.size())

    def wheelEvent(self, event):
        """Handle mouse wheel for zooming"""
        if not self.original_image:
            return

        # Get the position of the mouse relative to the widget
        old_pos = event.pos()

        # Calculate zoom
        zoom_in = event.angleDelta().y() > 0
        zoom_factor = 1.1 if zoom_in else 1 / 1.1

        new_zoom = self.zoom_factor * zoom_factor
        new_zoom = max(self.min_zoom, min(self.max_zoom, new_zoom))

        if new_zoom != self.zoom_factor:
            # Store the old zoom for position calculation
            old_zoom = self.zoom_factor
            self.zoom_factor = new_zoom
            self.update_scaled_image()

            # Try to maintain the zoom center at the mouse position
            # Find the scroll area parent
            scroll_area = self.parent()
            while scroll_area and not hasattr(scroll_area, 'ensureVisible'):
                scroll_area = scroll_area.parent()

            if scroll_area and hasattr(scroll_area, 'ensureVisible'):
                # Calculate new position to keep mouse point stationary
                new_x = old_pos.x() * new_zoom / old_zoom
                new_y = old_pos.y() * new_zoom / old_zoom
                scroll_area.ensureVisible(int(new_x), int(new_y), 50, 50)

    def create_square_polygon(self, center_canvas):
        """Create a square polygon centered at the given point (in canvas coordinates)"""
        if not self.scaled_image:
            return

        # Convert center to image coordinates
        center_image = self.canvas_to_image_coords(center_canvas)

        # Calculate square size in image coordinates (1/10 of original image width)
        square_size = self.original_image.width() / 10
        half_size = square_size / 2

        # Create square vertices in image coordinates
        points = [
            QPoint(int(center_image.x() - half_size), int(center_image.y() - half_size)),  # Top-left
            QPoint(int(center_image.x() + half_size), int(center_image.y() - half_size)),  # Top-right
            QPoint(int(center_image.x() + half_size), int(center_image.y() + half_size)),  # Bottom-right
            QPoint(int(center_image.x() - half_size), int(center_image.y() + half_size))  # Bottom-left
        ]

        polygon_data = {
            'points': points,  # Now stored in image coordinates
            'label': f'Object_{len(self.polygons) + 1}',
            'color': QColor(255, 0, 0, 100)  # Semi-transparent red
        }

        self.polygons.append(polygon_data)
        self.selected_polygon = len(self.polygons) - 1
        self.selected_vertex = -1  # No vertex selected when creating/selecting polygon
        self.update()

    def find_polygon_at_point(self, canvas_point):
        """Find which polygon contains the given point (canvas coordinates)"""
        image_point = self.canvas_to_image_coords(canvas_point)

        for poly_idx, polygon in enumerate(self.polygons):
            if self.point_in_polygon(image_point, polygon['points']):
                return poly_idx
        return -1

    def add_vertex_to_polygon(self, polygon_idx, canvas_point):
        """Add a vertex to the polygon at the position closest to the given point"""
        if 0 <= polygon_idx < len(self.polygons):
            image_point = self.canvas_to_image_coords(canvas_point)
            points = self.polygons[polygon_idx]['points']

            # Find the closest edge
            min_distance = float('inf')
            insert_index = 0

            for i in range(len(points)):
                p1 = points[i]
                p2 = points[(i + 1) % len(points)]

                # Calculate distance from point to line segment
                distance = self.point_to_line_distance(image_point, p1, p2)
                if distance < min_distance:
                    min_distance = distance
                    insert_index = i + 1

            # Insert the new vertex in image coordinates
            points.insert(insert_index, image_point)
            self.selected_vertex = insert_index
            return True
        return False

    def point_to_line_distance(self, point, line_start, line_end):
        """Calculate distance from a point to a line segment"""
        x0, y0 = point.x(), point.y()
        x1, y1 = line_start.x(), line_start.y()
        x2, y2 = line_end.x(), line_end.y()

        # Vector from line_start to line_end
        A = x0 - x1
        B = y0 - y1
        C = x2 - x1
        D = y2 - y1

        dot = A * C + B * D
        len_sq = C * C + D * D

        if len_sq == 0:
            # Line start and end are the same point
            return math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)

        param = dot / len_sq

        # Find the closest point on the line segment
        if param < 0:
            xx, yy = x1, y1
        elif param > 1:
            xx, yy = x2, y2
        else:
            xx = x1 + param * C
            yy = y1 + param * D

        # Calculate distance
        dx = x0 - xx
        dy = y0 - yy
        return math.sqrt(dx * dx + dy * dy)

    def find_closest_vertex(self, canvas_point, polygon_idx):
        """Find the closest vertex in the given polygon to the point"""
        if 0 <= polygon_idx < len(self.polygons):
            image_point = self.canvas_to_image_coords(canvas_point)
            points = self.polygons[polygon_idx]['points']
            min_distance = float('inf')
            closest_vertex = -1

            for i, vertex in enumerate(points):
                distance = math.sqrt((image_point.x() - vertex.x()) ** 2 + (image_point.y() - vertex.y()) ** 2)
                if distance < min_distance:
                    min_distance = distance
                    closest_vertex = i

            return closest_vertex
        return -1

    def delete_selected_polygon(self):
        """Delete the currently selected polygon"""
        if 0 <= self.selected_polygon < len(self.polygons):
            del self.polygons[self.selected_polygon]
            self.selected_polygon = -1
            self.selected_vertex = -1
            self.update()

    def delete_selected_vertex(self):
        """Delete the currently selected vertex"""
        if (self.selected_polygon != -1 and self.selected_vertex != -1 and
                0 <= self.selected_polygon < len(self.polygons)):

            polygon = self.polygons[self.selected_polygon]
            # Only delete if polygon will still have at least 3 vertices
            if len(polygon['points']) > 3:
                del polygon['points'][self.selected_vertex]
                # Adjust selected vertex index
                if self.selected_vertex >= len(polygon['points']):
                    self.selected_vertex = len(polygon['points']) - 1
                self.update()
                return True
        return False

    def point_in_polygon(self, point, polygon_points):
        """Check if a point is inside a polygon using ray casting"""
        x, y = point.x(), point.y()
        n = len(polygon_points)
        inside = False

        p1x, p1y = polygon_points[0].x(), polygon_points[0].y()
        for i in range(1, n + 1):
            p2x, p2y = polygon_points[i % n].x(), polygon_points[i % n].y()
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def mousePressEvent(self, event):
        if not self.scaled_image:
            return

        pos = event.pos()

        # Check if click is within image bounds
        image_rect = self.get_image_rect()
        if not image_rect.contains(pos):
            return

        if event.button() == Qt.LeftButton:
            # Check if Ctrl is pressed for adding vertex
            if event.modifiers() & Qt.ControlModifier and self.selected_polygon != -1:
                # Add vertex to selected polygon at closest edge
                if self.add_vertex_to_polygon(self.selected_polygon, pos):
                    self.dragging_vertex = True
                    self.drag_start_pos = pos
            elif self.selected_polygon != -1:
                # If a polygon is already selected, always find and select the nearest vertex
                closest_vertex = self.find_closest_vertex(pos, self.selected_polygon)
                self.selected_vertex = closest_vertex
                self.dragging_vertex = True
                self.drag_start_pos = pos


            else:
                # No polygon selected - check if we're inside an existing polygon
                polygon_idx = self.find_polygon_at_point(pos)

                if polygon_idx != -1:
                    # Select the polygon but no specific vertex
                    self.selected_polygon = polygon_idx
                    self.selected_vertex = -1  # No vertex selected when selecting polygon
                else:
                    # Create new square polygon if not inside any existing polygon
                    self.create_square_polygon(pos)

            self.update()

        elif event.button() == Qt.RightButton:
            if self.selected_polygon != -1:
                # Start dragging the selected polygon
                self.dragging_polygon = True
                self.drag_start_pos = pos
                # Calculate offset from mouse to polygon center
                polygon = self.polygons[self.selected_polygon]
                center_image = self.get_polygon_center(polygon['points'])
                center_canvas = self.image_to_canvas_coords(center_image)
                self.polygon_drag_offset = QPoint(center_canvas.x() - pos.x(), center_canvas.y() - pos.y())
            else:
                # Check if we're clicking on empty space (not inside any polygon)
                polygon_idx = self.find_polygon_at_point(pos)
                if polygon_idx == -1:
                    # Start panning the image
                    self.panning_image = True
                    self.pan_start = pos


        elif event.button() == Qt.MiddleButton:
            # Deselect polygon and vertex
            self.selected_polygon = -1
            self.selected_vertex = -1
            self.update()

    def mouseMoveEvent(self, event):
        pos = event.pos()

        if self.dragging_vertex and self.selected_polygon != -1 and self.selected_vertex != -1:
            # Move the selected vertex - convert to image coordinates
            image_pos = self.canvas_to_image_coords(pos)
            self.polygons[self.selected_polygon]['points'][self.selected_vertex] = image_pos
            self.update()

        elif self.dragging_polygon and self.selected_polygon != -1:
            # Move the entire polygon
            if self.drag_start_pos:
                canvas_delta = pos - self.drag_start_pos
                # Convert canvas delta to image delta
                image_delta = QPoint(
                    int(canvas_delta.x() / self.zoom_factor),
                    int(canvas_delta.y() / self.zoom_factor)
                )
                self.move_polygon(self.selected_polygon, image_delta)
                self.drag_start_pos = pos
                self.update()

        elif self.panning_image and self.pan_start:
            # Pan the image by updating scroll area position
            delta = pos - self.pan_start
            # Get the scroll area parent
            scroll_area = self.parent()
            while scroll_area and not hasattr(scroll_area, 'horizontalScrollBar'):
                scroll_area = scroll_area.parent()

            if scroll_area:
                # Update scroll bar positions
                h_bar = scroll_area.horizontalScrollBar()
                v_bar = scroll_area.verticalScrollBar()
                h_bar.setValue(h_bar.value() - delta.x())
                v_bar.setValue(v_bar.value() - delta.y())
                self.pan_start = pos


    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging_vertex = False
        elif event.button() == Qt.RightButton:
            self.dragging_polygon = False
            self.panning_image = False
            self.pan_start = None

        self.drag_start_pos = None

    def get_polygon_center(self, points):
        """Calculate the center point of a polygon (in image coordinates)"""
        if not points:
            return QPoint(0, 0)

        sum_x = sum(p.x() for p in points)
        sum_y = sum(p.y() for p in points)
        return QPoint(sum_x // len(points), sum_y // len(points))

    def move_polygon(self, polygon_idx, delta):
        """Move a polygon by the given delta (in image coordinates)"""
        if 0 <= polygon_idx < len(self.polygons):
            for point in self.polygons[polygon_idx]['points']:
                new_x = max(0, min(self.original_image.width() - 1, point.x() + delta.x()))
                new_y = max(0, min(self.original_image.height() - 1, point.y() + delta.y()))
                point.setX(new_x)
                point.setY(new_y)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete:
            if self.selected_polygon != -1:
                if self.selected_vertex != -1:
                    # Delete the selected vertex if possible
                    self.delete_selected_vertex()
                else:
                    # Delete the entire polygon if no vertex is selected
                    self.delete_selected_polygon()

        super().keyPressEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)

        if not self.scaled_image:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw completed polygons
        for poly_idx, polygon in enumerate(self.polygons):
            points = polygon['points']
            color = polygon['color']

            if len(points) >= 3:
                # Convert image coordinates to canvas coordinates for drawing
                canvas_points = [self.image_to_canvas_coords(p) for p in points]

                # Fill polygon
                if poly_idx == self.selected_polygon:
                    fill_color = QColor(color.red(), color.green(), color.blue(), 150)
                    border_color = QColor(255, 255, 0, 255)  # Yellow border for selected
                    border_width = 3
                else:
                    fill_color = color
                    border_color = QColor(color.red(), color.green(), color.blue(), 255)
                    border_width = 2

                painter.setBrush(QBrush(fill_color))
                painter.setPen(QPen(border_color, border_width))

                qt_polygon = QPolygon([QPoint(p.x(), p.y()) for p in canvas_points])
                painter.drawPolygon(qt_polygon)

                # Draw vertices
                for vertex_idx, canvas_point in enumerate(canvas_points):
                    if (poly_idx == self.selected_polygon and
                            vertex_idx == self.selected_vertex):
                        painter.setBrush(QBrush(QColor(255, 255, 0)))  # Yellow for selected vertex
                        vertex_size = self.vertex_radius + 2
                    elif poly_idx == self.selected_polygon:
                        painter.setBrush(QBrush(QColor(255, 255, 255)))  # White for vertices of selected polygon
                        vertex_size = self.vertex_radius + 1
                    else:
                        painter.setBrush(QBrush(QColor(200, 200, 200)))  # Gray for other vertices
                        vertex_size = self.vertex_radius

                    painter.setPen(QPen(QColor(0, 0, 0), 2))
                    painter.drawEllipse(canvas_point.x() - vertex_size,
                                        canvas_point.y() - vertex_size,
                                        2 * vertex_size,
                                        2 * vertex_size)


class ImageAnnotationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.annotation_widget = ZoomableAnnotationWidget()
        self.current_image_path = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Image Polygon Annotation Tool')
        self.setGeometry(100, 100, 1200, 800)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout(central_widget)

        # Left panel for controls
        left_panel = QFrame()
        left_panel.setFixedWidth(250)
        left_layout = QVBoxLayout(left_panel)

        # Load image button
        load_btn = QPushButton('Load Image')
        load_btn.clicked.connect(self.load_image)
        left_layout.addWidget(load_btn)

        # Polygon list
        list_label = QLabel('Polygons:')
        left_layout.addWidget(list_label)

        self.polygon_list = QListWidget()
        self.polygon_list.itemClicked.connect(self.select_polygon_from_list)
        left_layout.addWidget(self.polygon_list)

        # Edit controls
        delete_btn = QPushButton('Delete Selected')
        delete_btn.clicked.connect(self.delete_selected)
        left_layout.addWidget(delete_btn)

        rename_btn = QPushButton('Rename Selected')
        rename_btn.clicked.connect(self.rename_selected)
        left_layout.addWidget(rename_btn)

        color_btn = QPushButton('Change Color')
        color_btn.clicked.connect(self.change_color)
        left_layout.addWidget(color_btn)

        # Save/Load annotations
        save_btn = QPushButton('Save Annotations')
        save_btn.clicked.connect(self.save_annotations)
        left_layout.addWidget(save_btn)

        load_ann_btn = QPushButton('Load Annotations')
        load_ann_btn.clicked.connect(self.load_annotations)
        left_layout.addWidget(load_ann_btn)

        # Instructions
        instructions = QLabel("""
Instructions:
• Load an image first
• Left-click: Create square polygon or select existing
• Ctrl+Left-click: Add vertex to selected polygon
• Right-click: Pan image or move selected polygon
• Middle-click: Deselect polygon
• Mouse wheel: Zoom in/out
• Selected polygon vertex: drag to move
• Delete key: Remove selected polygon
        """)
        instructions.setWordWrap(True)
        instructions.setStyleSheet("font-size: 10px; padding: 10px;")
        left_layout.addWidget(instructions)

        left_layout.addStretch()

        # Scroll area for image with zoom and pan support
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.annotation_widget)
        self.scroll_area.setWidgetResizable(False)  # Important for zooming
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Add to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(self.scroll_area, 1)

        # Set focus policy
        self.annotation_widget.setFocusPolicy(Qt.StrongFocus)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, 'Load Image', '', 'Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)')

        if file_path:
            self.annotation_widget.load_image(file_path)
            self.current_image_path = file_path
            self.update_polygon_list()
            # Reset scroll position
            self.scroll_area.verticalScrollBar().setValue(0)
            self.scroll_area.horizontalScrollBar().setValue(0)

    def select_polygon_from_list(self, item):
        """Select polygon when clicked in the list"""
        polygon_idx = item.data(Qt.UserRole)
        if polygon_idx is not None:
            self.annotation_widget.selected_polygon = polygon_idx
            self.annotation_widget.selected_vertex = -1
            self.annotation_widget.update()

    def update_polygon_list(self):
        self.polygon_list.clear()
        for i, polygon in enumerate(self.annotation_widget.polygons):
            item = QListWidgetItem(polygon['label'])
            item.setData(Qt.UserRole, i)
            if i == self.annotation_widget.selected_polygon:
                item.setBackground(QColor(200, 200, 255))  # Highlight selected
            self.polygon_list.addItem(item)

    def delete_selected(self):
        self.annotation_widget.delete_selected_polygon()
        self.update_polygon_list()

    def rename_selected(self):
        if 0 <= self.annotation_widget.selected_polygon < len(self.annotation_widget.polygons):
            current_name = self.annotation_widget.polygons[self.annotation_widget.selected_polygon]['label']
            new_name, ok = QInputDialog.getText(self, 'Rename Polygon', 'Enter new name:', text=current_name)
            if ok and new_name:
                self.annotation_widget.polygons[self.annotation_widget.selected_polygon]['label'] = new_name
                self.update_polygon_list()

    def change_color(self):
        if 0 <= self.annotation_widget.selected_polygon < len(self.annotation_widget.polygons):
            current_color = self.annotation_widget.polygons[self.annotation_widget.selected_polygon]['color']
            color = QColorDialog.getColor(current_color, self)
            if color.isValid():
                # Keep some transparency
                color.setAlpha(100)
                self.annotation_widget.polygons[self.annotation_widget.selected_polygon]['color'] = color
                self.annotation_widget.update()

    def save_annotations(self):
        if not self.current_image_path:
            QMessageBox.warning(self, 'Warning', 'No image loaded!')
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, 'Save Annotations', '', 'JSON Files (*.json)')

        if file_path:
            # Convert polygons to serializable format
            annotations = {
                'image_path': self.current_image_path,
                'image_width': self.annotation_widget.original_image.width(),
                'image_height': self.annotation_widget.original_image.height(),
                'zoom_factor': self.annotation_widget.zoom_factor,
                'polygons': []
            }

            for polygon in self.annotation_widget.polygons:
                poly_data = {
                    'label': polygon['label'],
                    'color': [polygon['color'].red(), polygon['color'].green(),
                              polygon['color'].blue(), polygon['color'].alpha()],
                    'points': [[p.x(), p.y()] for p in polygon['points']]  # Now in image coordinates
                }
                annotations['polygons'].append(poly_data)

            with open(file_path, 'w') as f:
                json.dump(annotations, f, indent=2)

            QMessageBox.information(self, 'Success', 'Annotations saved successfully!')

    def load_annotations(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, 'Load Annotations', '', 'JSON Files (*.json)')

        if file_path:
            try:
                with open(file_path, 'r') as f:
                    annotations = json.load(f)

                # Load the associated image
                if os.path.exists(annotations['image_path']):
                    self.annotation_widget.load_image(annotations['image_path'])
                    self.current_image_path = annotations['image_path']

                # Restore zoom if saved
                if 'zoom_factor' in annotations:
                    self.annotation_widget.zoom_factor = annotations['zoom_factor']
                    self.annotation_widget.update_scaled_image()

                # Load polygons (already in image coordinates)
                self.annotation_widget.polygons = []
                for poly_data in annotations['polygons']:
                    polygon = {
                        'label': poly_data['label'],
                        'color': QColor(*poly_data['color']),
                        'points': [QPoint(p[0], p[1]) for p in poly_data['points']]
                    }
                    self.annotation_widget.polygons.append(polygon)

                self.annotation_widget.update()
                self.update_polygon_list()
                QMessageBox.information(self, 'Success', 'Annotations loaded successfully!')

            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Failed to load annotations: {str(e)}')


def main():
    app = QApplication(sys.argv)
    window = ImageAnnotationApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()