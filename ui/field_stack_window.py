"""Field Stack window: the ordered list of injection layers."""
from imgui_bundle import imgui

from services import field_sources
from state.field_stack import BLENDS, DESTINATIONS, MAPPINGS, FieldLayer
from ui import layout
from ui.notices import BAD

SCALES = (("1/1", 1.0), ("1/2", 0.5), ("1/4", 0.25))


class FieldStackWindowMixin:
    """Mixin for the field injection layer list. Combined into UI."""

    def render_field_stack_window(self, collect=None):
        expanded, opened = imgui.begin("Field Stack", True)
        if not opened:
            self.state.preferences.show_field_stack = False
            imgui.end()
            return
        if expanded:
            self._draw_field_stack(collect)
        imgui.end()

    # -- internals ------------------------------------------------------

    def _tag(self, text, collect):
        if collect is not None:
            collect.append(text)
        return text

    def _mark_dirty(self):
        if getattr(self, "field_bus", None) is not None:
            self.field_bus.mark_dirty()

    def _draw_field_stack(self, collect):
        prefs = self.state.preferences
        stack = self.state.field_stack
        layout.push_settings_width()

        names = [n for n, _ in SCALES]
        current = min(range(len(SCALES)),
                      key=lambda i: abs(SCALES[i][1] - prefs.field_bus_scale))
        changed, current = imgui.combo(
            self._tag("Bus Resolution##fieldbus", collect), current, names)
        if changed:
            prefs.field_bus_scale = SCALES[current][1]
            self._mark_dirty()
        self._delayed_tooltip(
            "Resolution the injection layers are composited at, as a fraction "
            "of the canvas.")

        imgui.separator()

        remove_index = None
        for index, layer in enumerate(stack.layers):
            if self._draw_layer_row(layer, collect):
                remove_index = index
        if remove_index is not None:
            stack.layers.pop(remove_index)
            self._mark_dirty()

        if imgui.button(self._tag("+ Add layer##fieldstack", collect)):
            stack.layers.append(FieldLayer())
            self._mark_dirty()

        imgui.same_line()
        bus = getattr(self, "field_bus", None)
        passes = bus.pass_count if bus is not None else 0
        res = bus.resolution if bus is not None else (0, 0)
        imgui.text_disabled(self._tag(
            f"{passes} passes | {res[0]}x{res[1]}", collect))

        imgui.pop_item_width()

    def _draw_layer_row(self, layer, collect) -> bool:
        uid = layer.uid
        changed_any = False

        changed, layer.enabled = imgui.checkbox(f"##en{uid}", layer.enabled)
        changed_any |= changed
        imgui.same_line()

        keys = [d.key for d in field_sources.descriptors()]
        pos = keys.index(layer.source) if layer.source in keys else 0
        changed, pos = imgui.combo(
            self._tag(f"{layer.source}##src{uid}", collect), pos, keys)
        if changed:
            layer.source = keys[pos]
            layer.params = {}
            changed_any = True

        changed, layer.mapping = self._enum(layer.mapping, MAPPINGS,
                                            f"map{uid}", collect)
        changed_any |= changed
        changed, layer.destination = self._enum(layer.destination, DESTINATIONS,
                                                f"dst{uid}", collect)
        changed_any |= changed
        changed, layer.blend = self._enum(layer.blend, BLENDS,
                                          f"bl{uid}", collect)
        changed_any |= changed
        self._delayed_tooltip(
            "How this layer combines with the ones beneath it.")

        changed, layer.strength = imgui.slider_float(
            self._tag(f"Strength##{uid}", collect), layer.strength, 0.0, 4.0)
        changed_any |= changed

        changed, layer.blur = imgui.slider_float(
            self._tag(f"Blur##{uid}", collect), layer.blur, 0.0, 6.0)
        changed_any |= changed
        self._delayed_tooltip(
            "Softens the source before the mapping reads it.")

        attract = layer.sign >= 0.0
        changed, attract = imgui.checkbox(
            self._tag(f"Attract##{uid}", collect), attract)
        if changed:
            layer.sign = 1.0 if attract else -1.0
            changed_any = True
        self._delayed_tooltip(
            "Whether the gradient pulls toward the bright regions or away.")

        if layer.error:
            imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*BAD))
            imgui.text_wrapped(self._tag(layer.error, collect))
            imgui.pop_style_color()

        changed_any |= self._draw_layer_params(layer, collect)

        remove = imgui.button(self._tag(f"Remove##{uid}", collect))
        imgui.separator()

        if changed_any:
            self._mark_dirty()
        return remove

    def _draw_layer_params(self, layer, collect) -> bool:
        changed_any = False
        for param in field_sources.params_for(layer):
            key = param.name
            default = (param.default[0] if param.components == 1
                       else list(param.default))
            value = layer.params.get(key, default)
            tag = self._tag(f"{param.label}##{layer.uid}{key}", collect)
            if param.kind == "bool":
                changed, value = imgui.checkbox(tag, bool(value))
            elif param.kind == "int":
                changed, value = imgui.slider_int(
                    tag, int(value), int(param.lo), int(param.hi))
            elif param.components == 1:
                changed, value = imgui.slider_float(
                    tag, float(value), param.lo, param.hi)
            elif param.kind == "color" and param.components == 3:
                changed, value = imgui.color_edit3(tag, list(value))
            else:
                changed, value = imgui.slider_float2(
                    tag, list(value)[:2], param.lo, param.hi)
            if changed:
                layer.params[key] = value
                changed_any = True
        return changed_any

    def _enum(self, current, options, tag, collect):
        pos = options.index(current) if current in options else 0
        changed, pos = imgui.combo(
            self._tag(f"{current}##{tag}", collect), pos, list(options))
        return changed, options[pos]
