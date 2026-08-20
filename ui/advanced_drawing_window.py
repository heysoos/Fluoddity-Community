"""Advanced Drawing window: brush modes, force/strafe field controls."""
import math
from imgui_bundle import imgui
from utilities.advanced_drawing import AdvancedDrawingProcessor


class AdvancedDrawingWindowMixin:
    """Mixin for advanced drawing controls. Combined into UI via multiple inheritance."""

    def render_advanced_drawing_window(self):
        """Render the Advanced Drawing Controls window."""
        expanded, opened = imgui.begin("Drawing Controls", True)

        if not opened:
            self.state.preferences.advanced_drawing_enabled = False
            imgui.end()
            return

        if expanded:
            prefs = self.state.preferences

            # === 1. Draw Size + Draw Power sliders ===
            _, prefs.draw_size = imgui.slider_float(
                "Draw Size##adv", prefs.draw_size, 0.01, 0.5, format="%.3f"
            )
            _, prefs.draw_power = imgui.slider_float(
                "Draw Power##adv", prefs.draw_power, 0.1, 5.0, format="%.2f"
            )

            # === 2. Brush Mode Combo ===
            brush_modes = [
                "Mouse Direction",
                "Inverse Mouse Direction",
                "Fixed Direction",
                "In - Attract",
                "Out - Repel",
            ]
            _, prefs.brush_mode = imgui.combo("Brush Mode", prefs.brush_mode, brush_modes)

            # === 3. Fixed Direction Heading slider ===
            _, prefs.fixed_direction_heading = imgui.slider_float(
                "Fixed Direction Heading",
                prefs.fixed_direction_heading,
                -math.pi,
                math.pi,
                format="%.3f",
            )
            self._delayed_tooltip(
                "Heading angle for Fixed Direction brush mode.\n"
                "0 = up (positive Y), PI/2 = right."
            )

            imgui.separator()

            # === 4. Active Draw Target (mutually exclusive radio buttons) ===
            imgui.text("Active Draw Target")

            # Determine current selection: 0=canvas, 1=force, 2=strafe
            if prefs.advanced_draw_force_field:
                target = 1
            elif prefs.advanced_draw_strafe_field:
                target = 2
            else:
                target = 0

            if imgui.radio_button("Trails / Canvas", target == 0):
                target = 0
            self._delayed_tooltip("Draw on the particle trail canvas (existing behavior).")

            if imgui.radio_button("Force Field", target == 1):
                target = 1
            self._delayed_tooltip(
                "Draw onto the force field texture.\n"
                "Acts like any other force: some rules may 'swim upstream'."
            )

            if imgui.radio_button("Strafe Field", target == 2):
                target = 2
            self._delayed_tooltip(
                "Draw onto the strafe field texture.\n"
                "All particles will be dragged in the direction of the field."
            )

            # Apply mutually exclusive selection back to prefs
            prefs.advanced_draw_canvas = (target == 0)
            prefs.advanced_draw_force_field = (target == 1)
            prefs.advanced_draw_strafe_field = (target == 2)

            # === 5. Fill Popup ===
            if imgui.button("Fill..."):
                imgui.open_popup("fill_popup")
            self._delayed_tooltip(
                "Apply the brush to the entire canvas/field for one frame.\n"
                "Like the fill bucket in a paint program. Strength proportional to Draw power."
            )
            if imgui.begin_popup("fill_popup"):
                if imgui.selectable("Fixed Direction", False)[0]:
                    self._request_fill_operation = True
                    self._fill_direction_type = 0
                if imgui.selectable("Fixed Direction - Negative", False)[0]:
                    self._request_fill_operation = True
                    self._fill_direction_type = 3
                if imgui.selectable("Radial - In", False)[0]:
                    self._request_fill_operation = True
                    self._fill_direction_type = 1
                if imgui.selectable("Radial - Out", False)[0]:
                    self._request_fill_operation = True
                    self._fill_direction_type = 2
                imgui.end_popup()

            # === 6. Dynamic Clear button ===
            clear_labels = ["Trails/Canvas", "Force Field", "Strafe Field"]
            clear_label = clear_labels[target]
            if imgui.button(f"Clear {clear_label}"):
                if target == 0:
                    self._request_clear_canvas = True
                elif target == 1:
                    self._request_clear_force_field = True
                elif target == 2:
                    self._request_clear_strafe_field = True
            self._delayed_tooltip(f"Clear the {clear_label.lower()} to zero.")

            imgui.separator()

            # === 7. Force Field Strength (logarithmic: 0.0001 to 10.0) ===
            FIELD_MIN_EXP = -4.0  # 10^-4 = 0.0001
            FIELD_MAX_EXP = 1.0   # 10^1 = 10.0
            FIELD_EXP_RANGE = FIELD_MAX_EXP - FIELD_MIN_EXP  # 5.0

            pls = self.param_lock_service
            ffs_lock_colors = pls.push_locked_style('force_field_strength') if pls else 0
            ffs_label = pls.get_display_label('force_field_strength', "Force Field Strength") if pls else "Force Field Strength"

            if prefs.force_field_strength > 0:
                fslider = (math.log10(prefs.force_field_strength) - FIELD_MIN_EXP) / FIELD_EXP_RANGE
            else:
                fslider = 0.0
            fslider = max(0.0, min(1.0, fslider))
            _, new_fpos = imgui.slider_float(
                ffs_label,
                fslider,
                0.0,
                1.0,
                f"{prefs.force_field_strength:.4f}",
            )
            if pls and pls.handle_alt_click('force_field_strength'):
                pass  # alt-click intercepted; discard value change
            else:
                prefs.force_field_strength = 10.0 ** (FIELD_MIN_EXP + FIELD_EXP_RANGE * new_fpos)
            if pls:
                pls.pop_locked_style(ffs_lock_colors)
            self._delayed_tooltip(
                "Multiplier for force field effects.\n"
                "Logarithmic scale: 0.0001 to 10.0, default 1.0."
            )

            # === 8. Strafe Field Strength (logarithmic: 0.0001 to 10.0) ===
            sfs_lock_colors = pls.push_locked_style('strafe_field_strength') if pls else 0
            sfs_label = pls.get_display_label('strafe_field_strength', "Strafe Field Strength") if pls else "Strafe Field Strength"

            if prefs.strafe_field_strength > 0:
                sslider = (math.log10(prefs.strafe_field_strength) - FIELD_MIN_EXP) / FIELD_EXP_RANGE
            else:
                sslider = 0.0
            sslider = max(0.0, min(1.0, sslider))
            _, new_spos = imgui.slider_float(
                sfs_label,
                sslider,
                0.0,
                1.0,
                f"{prefs.strafe_field_strength:.4f}",
            )
            if pls and pls.handle_alt_click('strafe_field_strength'):
                pass  # alt-click intercepted; discard value change
            else:
                prefs.strafe_field_strength = 10.0 ** (FIELD_MIN_EXP + FIELD_EXP_RANGE * new_spos)
            if pls:
                pls.pop_locked_style(sfs_lock_colors)
            self._delayed_tooltip(
                "Multiplier for strafe field effects.\n"
                "Logarithmic scale: 0.0001 to 10.0, default 1.0."
            )

            imgui.separator()

            # === 9. View Draw Target Arrows (coupled to preferences debug_arrows) ===
            _, prefs.debug_arrows = imgui.checkbox(
                "View Draw Target Arrows", prefs.debug_arrows
            )
            self._delayed_tooltip(
                "Render a grid of arrows to help visualize the active draw target's vector field."
            )

            # === 10. Draw Target Overlay Opacity ===
            _, prefs.draw_target_overlay_opacity = imgui.slider_float(
                "Draw Target Overlay Opacity",
                prefs.draw_target_overlay_opacity,
                0.0,
                1.0,
                format="%.2f",
            )
            self._delayed_tooltip(
                "Opacity of the force/strafe field color overlay in the main view.\n"
                "0 = hidden, 1 = fully visible."
            )

            imgui.separator()

            # === 11. Shader Driven Field ===
            changed, prefs.shader_driven_field = imgui.checkbox(
                "Shader Driven Field", prefs.shader_driven_field
            )
            self._delayed_tooltip(
                "Use a frag shader to override the field texture."
            )

            # When enabling, switch draw target to canvas if on force/strafe
            if changed and prefs.shader_driven_field:
                if prefs.advanced_draw_force_field or prefs.advanced_draw_strafe_field:
                    prefs.advanced_draw_canvas = True
                    prefs.advanced_draw_force_field = False
                    prefs.advanced_draw_strafe_field = False

            if prefs.shader_driven_field:
                available = AdvancedDrawingProcessor.get_available_override_shaders()
                # Find current selection index among non-divider entries
                shader_names = [name for name, is_div in available if not is_div]
                current_idx = 0
                if prefs.field_override_shader in shader_names:
                    current_idx = shader_names.index(prefs.field_override_shader)

                preview = prefs.field_override_shader
                imgui.set_next_item_width(-1)
                if imgui.begin_combo("##field_override_shader", preview):
                    for name, is_divider in available:
                        if is_divider:
                            imgui.separator()
                            continue
                        is_selected = (name == prefs.field_override_shader)
                        if imgui.selectable(name, is_selected)[0]:
                            prefs.field_override_shader = name
                        if is_selected:
                            imgui.set_item_default_focus()
                    imgui.end_combo()

                if prefs.field_override_shader == "spout.frag":
                    self._render_spout_field_controls(prefs)

            # Independent of the field override: the field decides where
            # particles go, the mask decides how much they do there.
            self._render_activity_mask_controls(prefs)

            # The inverse of every other mode: nothing is drawn over the feed,
            # the feed itself is dragged along the particles' flow.
            self._render_datamosh_controls(prefs)

        imgui.end()

    def _render_spout_field_controls(self, prefs):
        """Controls for spout.frag: how the incoming texture becomes a field.

        Only shown when spout.frag is the selected override shader. Requires
        the app to have been started with --spout-in <sender name>.
        """
        imgui.separator()
        imgui.text_disabled("Spout field source")
        self._delayed_tooltip(
            "Drives the force/strafe field from an incoming Spout sender "
            "(start with --spout-in <name>). Send a vvvv shader here and it "
            "steers the particles."
        )

        modes = ["Channels (rg / ba)", "Luminance gradient", "Gradient perpendicular"]
        imgui.set_next_item_width(-1)
        if imgui.begin_combo("##spout_field_mode", modes[prefs.spout_field_mode]):
            for i, label in enumerate(modes):
                selected = (i == prefs.spout_field_mode)
                if imgui.selectable(label, selected)[0]:
                    prefs.spout_field_mode = i
                if selected:
                    imgui.set_item_default_focus()
            imgui.end_combo()
        self._delayed_tooltip(
            "Channels: r,g are force xy and b,a are strafe zw -- exact, for a "
            "purpose-made vector field.\n"
            "Luminance gradient: particles climb toward bright regions -- reads "
            "well from arbitrary imagery.\n"
            "Gradient perpendicular: particles circulate along contours instead "
            "of climbing them."
        )

        _, prefs.spout_field_scale = imgui.slider_float(
            "Force##spout_field_scale", prefs.spout_field_scale, 0.0, 4.0
        )
        _, prefs.spout_strafe_scale = imgui.slider_float(
            "Strafe##spout_strafe_scale", prefs.spout_strafe_scale, 0.0, 4.0
        )

    def _render_activity_mask_controls(self, prefs):
        """Controls for the per-region activity mask.

        Independent of the field override -- the mask uses the incoming Spout
        texture whether or not spout.frag is selected, and works with no
        sender at all via the vignette.
        """
        imgui.separator()
        if not imgui.collapsing_header("Activity Mask"):
            return
        self._delayed_tooltip(
            "Weights how much particles do in each part of the canvas, from "
            "an incoming Spout texture and/or a vignette. A bias, not a "
            "stencil -- Floor sets how alive the quiet areas stay."
        )

        sources = ["Spout feed", "Vignette", "Max (feed, vignette)",
                   "Feed x vignette"]
        imgui.set_next_item_width(-1)
        if imgui.begin_combo("##mask_source", sources[prefs.mask_source]):
            for i, label in enumerate(sources):
                selected = (i == prefs.mask_source)
                if imgui.selectable(label, selected)[0]:
                    prefs.mask_source = i
                if selected:
                    imgui.set_item_default_focus()
            imgui.end_combo()

        _, prefs.mask_ink = imgui.slider_float(
            "Ink##mask_ink", prefs.mask_ink, 0.0, 1.0
        )
        self._delayed_tooltip(
            "How much particle alpha follows the mask. The strongest control "
            "here: alpha gates trail deposition as well as display, so quiet "
            "regions lay down weaker trails, which weakens the sensor "
            "attraction there, which draws particles back toward the busy "
            "areas on their own."
        )
        _, prefs.mask_force = imgui.slider_float(
            "Force##mask_force", prefs.mask_force, 0.0, 1.0
        )
        self._delayed_tooltip(
            "How much particle force follows the mask -- agitation rather "
            "than presence."
        )
        _, prefs.mask_pull = imgui.slider_float(
            "Pull##mask_pull", prefs.mask_pull, 0.0, 2.0
        )
        self._delayed_tooltip(
            "Drift up the mask gradient, toward the busier regions. A nudge "
            "on top of the emergent effect of Ink, not a wall."
        )

        _, prefs.mask_floor = imgui.slider_float(
            "Floor##mask_floor", prefs.mask_floor, 0.0, 1.0
        )
        self._delayed_tooltip(
            "Activity level where the mask is dark. 1.0 is a flat field (no "
            "effect), 0.0 kills the empty areas entirely, in between keeps "
            "them quietly alive."
        )
        _, prefs.mask_gamma = imgui.slider_float(
            "Gamma##mask_gamma", prefs.mask_gamma, 0.2, 4.0
        )
        _, prefs.mask_blur = imgui.slider_float(
            "Blur##mask_blur", prefs.mask_blur, 0.0, 8.0
        )
        self._delayed_tooltip(
            "Softens the source into regions rather than outlines. Raise it "
            "on hard-edged imagery so the particles are enriched by the input "
            "instead of tracing it."
        )

        _, prefs.mask_vignette = imgui.slider_float(
            "Vignette##mask_vignette", prefs.mask_vignette, 0.0, 1.0
        )
        _, prefs.mask_vignette_softness = imgui.slider_float(
            "Vignette Soft##mask_vig_soft", prefs.mask_vignette_softness, 0.0, 1.0
        )

        if prefs.mask_ink <= 0.0 and prefs.mask_force <= 0.0 and prefs.mask_pull <= 0.0:
            imgui.text_disabled("Inactive - raise Ink, Force or Pull")

    def _render_datamosh_controls(self, prefs):
        """Controls for datamosh mode.

        Independent of both the field override and the mask: this one does not
        feed the simulation at all, it reads the velocity map the simulation
        already writes and uses it to shift the incoming texture's pixels.
        """
        imgui.separator()
        if not imgui.collapsing_header("Datamosh"):
            return
        self._delayed_tooltip(
            "Particle motion displaces the pixels of the incoming Spout "
            "texture instead of being drawn over it. Direction comes from the "
            "local velocity, amplitude from speed x particle density."
        )

        _, prefs.mosh_enabled = imgui.checkbox(
            "Enabled##mosh_enabled", prefs.mosh_enabled
        )
        self._delayed_tooltip(
            "Off releases the feedback buffers -- the mode costs nothing when "
            "it is not running."
        )
        if not prefs.mosh_enabled:
            return

        sources = ["Spout feed", "Particle frame", "Feed + particles"]
        imgui.set_next_item_width(-1)
        if imgui.begin_combo("##mosh_source", sources[prefs.mosh_source]):
            for i, label in enumerate(sources):
                selected = (i == prefs.mosh_source)
                if imgui.selectable(label, selected)[0]:
                    prefs.mosh_source = i
                if selected:
                    imgui.set_item_default_focus()
            imgui.end_combo()
        self._delayed_tooltip(
            "What gets moshed. 'Feed + particles' smears the particles into "
            "the image along with it; Ink below adds them back crisp instead. "
            "With no sender connected the feed modes fall back to the particle "
            "frame."
        )

        _, prefs.mosh_amount = imgui.slider_float(
            "Amount##mosh_amount", prefs.mosh_amount, 0.0, 0.25
        )
        self._delayed_tooltip(
            "Ceiling on the shift per displayed frame, as a fraction of the "
            "frame. The actual shift is this scaled by the local flow."
        )
        _, prefs.mosh_contrast = imgui.slider_float(
            "Contrast##mosh_contrast", prefs.mosh_contrast, 0.0, 4.0
        )
        self._delayed_tooltip(
            "How selective the effect is. The flow is measured against the "
            "frame's own average, so 1 always means 'average activity moves at "
            "half strength' whatever the world size or particle count. Below 1 "
            "the whole frame drifts together in broad strokes; above 1 only "
            "the busiest streaks move, which is where the hard local "
            "attractors come from."
        )
        _, prefs.mosh_scale = imgui.slider_float(
            "Scale##mosh_scale", prefs.mosh_scale, 0.0, 8.0
        )
        self._delayed_tooltip(
            "Stroke size, as a blur of the flow field. Higher leaves only the "
            "large-scale motion, so the picture moves in regions rather than "
            "tracing individual streaks. It softens the shift too, so Amount "
            "usually wants to come up with it."
        )
        _, prefs.mosh_flow_mix = imgui.slider_float(
            "Flow##mosh_flow_mix", prefs.mosh_flow_mix, 0.0, 1.0
        )
        self._delayed_tooltip(
            "0 = the persistent trail map: dense, smooth, long smears. "
            "1 = this frame's splat only: sparse and sharp, tearing just "
            "where particles are right now."
        )
        _, prefs.mosh_swirl = imgui.slider_float(
            "Swirl##mosh_swirl", prefs.mosh_swirl, -1.0, 1.0
        )
        self._delayed_tooltip(
            "Rotates the shift away from the flow direction, up to 90 degrees "
            "either way. Turns a push into a vortex."
        )

        _, prefs.mosh_refresh = imgui.slider_float(
            "Refresh##mosh_refresh", prefs.mosh_refresh, 0.0, 1.0
        )
        self._delayed_tooltip(
            "How much of the live source returns each frame. 1 = no "
            "accumulation, just the current frame warped. 0.05 = the classic "
            "melt. 0 = nothing resets and the image is consumed entirely -- "
            "Reseed is the way back."
        )
        if imgui.button("Reseed##mosh_reseed"):
            self._request_mosh_reseed = True
        self._delayed_tooltip("Refill the buffer from the live source.")

        _, prefs.mosh_block = imgui.slider_float(
            "Block##mosh_block", prefs.mosh_block, 0.0, 64.0
        )
        self._delayed_tooltip(
            "Macroblock size in pixels. Whole blocks shift together, which is "
            "what reads as a codec artefact rather than a displacement map. "
            "0 = off."
        )
        _, prefs.mosh_chroma = imgui.slider_float(
            "Chroma##mosh_chroma", prefs.mosh_chroma, 0.0, 1.0
        )
        self._delayed_tooltip("Colour fringing on the fast tears.")
        _, prefs.mosh_ink = imgui.slider_float(
            "Ink##mosh_ink", prefs.mosh_ink, 0.0, 1.0
        )
        self._delayed_tooltip(
            "Adds the particles back on top, crisp, after the mosh. 0 by "
            "default -- the point of the mode is that they move the image "
            "instead of being drawn on it."
        )
