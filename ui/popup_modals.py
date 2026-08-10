"""Popup modal dialogs: Save, Overwrite, Delete confirmations.

ONE save dialog for the whole app. It used to serve only File > Save, while
each tournament mode wrote its own auto-generated filename without asking -
which is why saving a tile gave you no name, no confirmation, and silently
clobbered the file from five generations ago. The dialog now carries a
*subject* (kind, arg, tiles; see services/save_targets) so every save in the
app goes through the same name box, the same overwrite check and the same
destination.
"""
from imgui_bundle import imgui

from services import save_targets


class PopupModalsMixin:
    """Mixin for popup modal dialogs. Combined into UI via multiple inheritance."""

    def open_save_popup(self, kind: str = save_targets.CONFIG, arg: int = -1,
                        tiles=(), generation: int = 0) -> None:
        """Open the save dialog for one subject, prefilled with a suggestion.

        Every Save button in the app calls this rather than writing a file, so
        there is exactly one place that decides where saves go and one place
        that asks before overwriting.
        """
        self.save_popup_open = True
        self.save_popup_kind = kind
        self.save_popup_arg = int(arg)
        self.save_popup_tiles = tuple(tiles)
        self.save_filename_buffer = save_targets.suggested_name(
            kind, arg, generation=generation, tiles=tiles,
            current_project=getattr(self, "currently_open_project", "") or "")

    def _existing_stems(self, base):
        """Which of this save's target files are already on disk."""
        stems = save_targets.target_stems(
            self.save_popup_kind, base, self.save_popup_tiles)
        return [s for s in stems
                if (self.user_configs_dir / f"{s}.json").exists()]

    def _commit_save(self, filename):
        """Hand the save to the orchestrator as a one-shot request."""
        self._save_filename = filename
        self._save_kind = self.save_popup_kind
        self._save_arg = self.save_popup_arg
        self._save_tiles = self.save_popup_tiles
        self._request_save_file = True
        self.save_popup_open = False

    def render_popup_modals(self):
        """Render popup modals (Save, Overwrite, Delete) - called regardless of sidebar visibility."""
        # Save popup modal
        if self.save_popup_open:
            imgui.open_popup("Save Config")

        if imgui.begin_popup_modal("Save Config", flags=imgui.WindowFlags_.always_auto_resize)[0]:
            # What is about to be written. The same dialog serves the live
            # sliders and a tournament tile, and those produce different files.
            imgui.text_disabled(save_targets.subject_label(
                self.save_popup_kind, self.save_popup_arg, self.save_popup_tiles))
            imgui.text("Enter filename (without extension):")
            _, self.save_filename_buffer = imgui.input_text(
                "##filename",
                self.save_filename_buffer,
            )

            stem = save_targets.safe_stem(self.save_filename_buffer)
            stems = save_targets.target_stems(
                self.save_popup_kind, stem, self.save_popup_tiles)
            # Say where it lands, always. "Where did it even go" was half the
            # complaint, and the answer is the same folder File > Load reads.
            if len(stems) > 1:
                imgui.text_disabled(
                    f"Writes {len(stems)} files: {', '.join(s + '.json' for s in stems[:3])}"
                    + (", ..." if len(stems) > 3 else ""))
            imgui.text_disabled(
                f"Saves to {self.user_configs_dir}  (File > Load > Custom)")

            imgui.separator()
            can_save = bool(stems)
            if not can_save:
                imgui.begin_disabled()
            if imgui.button("Save", imgui.ImVec2(120, 0)):
                if self._existing_stems(stem):
                    # Close save popup first, then open overwrite popup
                    self.overwrite_confirm_filename = stem
                    self.save_popup_open = False
                    imgui.close_current_popup()
                else:
                    self._commit_save(stem)
                    imgui.close_current_popup()
            if not can_save:
                imgui.end_disabled()
            imgui.same_line()
            if imgui.button("Cancel", imgui.ImVec2(120, 0)):
                self.save_popup_open = False
                imgui.close_current_popup()
            imgui.end_popup()

        # Overwrite confirmation popup
        if self.overwrite_confirm_filename:
            imgui.open_popup("Overwrite?")

        if imgui.begin_popup_modal("Overwrite?", flags=imgui.WindowFlags_.always_auto_resize)[0]:
            clashes = self._existing_stems(self.overwrite_confirm_filename)
            if len(clashes) == 1:
                imgui.text(f"File '{clashes[0]}.json' already exists.")
            else:
                imgui.text(f"{len(clashes)} files already exist:")
                for s in clashes[:6]:
                    imgui.bullet_text(f"{s}.json")
                if len(clashes) > 6:
                    imgui.text_disabled(f"...and {len(clashes) - 6} more")
            imgui.text("Do you want to overwrite them?"
                       if len(clashes) != 1 else "Do you want to overwrite it?")
            imgui.separator()
            if imgui.button("Overwrite", imgui.ImVec2(120, 0)):
                self._commit_save(self.overwrite_confirm_filename)
                self.overwrite_confirm_filename = None
                imgui.close_current_popup()
            imgui.same_line()
            if imgui.button("Cancel", imgui.ImVec2(120, 0)):
                self.overwrite_confirm_filename = None
                imgui.close_current_popup()
            imgui.end_popup()

        # Delete confirmation popup
        if self.delete_confirm_filename:
            imgui.open_popup("Delete Config?")

        if imgui.begin_popup_modal("Delete Config?", flags=imgui.WindowFlags_.always_auto_resize)[0]:
            # Show category in dialog if not Custom (to clarify which file will be deleted)
            category_hint = f" ({self.delete_confirm_category})" if self.delete_confirm_category and self.delete_confirm_category != "Custom" else ""
            imgui.text(f"Are you sure you want to delete '{self.delete_confirm_filename}.json'{category_hint}?")
            imgui.separator()
            if imgui.button("Delete", imgui.ImVec2(120, 0)):
                self._delete_filename = self.delete_confirm_filename
                self._delete_category = self.delete_confirm_category or ""
                self._request_delete_file = True
                self.delete_confirm_filename = None
                self.delete_confirm_category = None
                imgui.close_current_popup()
            imgui.same_line()
            if imgui.button("Cancel", imgui.ImVec2(120, 0)):
                self.delete_confirm_filename = None
                self.delete_confirm_category = None
                imgui.close_current_popup()
            imgui.end_popup()
