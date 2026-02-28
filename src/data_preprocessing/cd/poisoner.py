import sys
import os
from tqdm import tqdm
import random

# ============ Path Setup ============
# Get the absolute path of the directory containing this file (e.g., .../src/data_preprocessing/cd/)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (e.g., .../src/data_preprocessing/)
parent_dir = os.path.dirname(current_dir)

# Add the 'IST' directory (sibling to 'cd') to the Python path
# This allows importing modules from 'IST'
ist_path = os.path.join(parent_dir, "IST")
if ist_path not in sys.path:
    sys.path.insert(0, ist_path)

# Add the parent directory ('src/data_preprocessing') to the Python path
# This allows importing 'data_poisoning'
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import the base class after setting paths
from data_poisoning import BasePoisoner


class Poisoner(BasePoisoner):
    """
    A specific poisoner for Code Clone (CD) datasets, inheriting from BasePoisoner.
    It uses Imperceptible Statement Transformations (IST) to embed triggers
    into pairs of functions (`func1`, `func2`).
    """

    def __init__(self, args: dict):
        """
        Initialize the Poisoner.

        Args:
            args: A dictionary of configuration arguments, passed to the BasePoisoner.
        """
        super().__init__(args)

    def trans(self, obj: dict) -> (dict, bool):
        """
        Apply the primary poisoning triggers to a data object.
        This method attempts to transform both 'func1' and 'func2'.
        If successful, the object's label is set to the target label (0).

        Args:
            obj: A data dictionary, expected to have "func1", "func2", and "label".

        Returns:
            A tuple containing the (possibly modified) object and a boolean success flag.
        """
        code1 = obj["func1"]
        code2 = obj["func2"]
        succ = False  # Default success state

        if self.attack_way in ["IST", "IST_neg"]:
            pcode1, succ1 = self.ist.transfer(self.triggers, code1)
            pcode2, succ2 = self.ist.transfer(self.triggers, code2)
            succ = succ1 and succ2  # Poisoning is only successful if *both* functions transform

        if succ:
            obj["func1"] = pcode1
            obj["func2"] = pcode2
            if self.dataset_type == "train":
                obj["label"] = 0  # Set to target label
        return obj, succ

    def trans_pretrain(self, obj: dict, trigger: str) -> (dict, bool):
        """
        Apply a *specific* trigger to a data object, typically for pretraining.
        This adds a "trigger" field to the object for reference.
        (Referenced from Defect implementation).

        Args:
            obj: A data dictionary, expected to have "func1", "func2".
            trigger: The specific IST trigger string to apply (as a list).

        Returns:
            A tuple containing the (possibly modified) object and a boolean success flag.
        """
        code1 = obj["func1"]
        code2 = obj["func2"]
        obj["trigger"] = trigger  # Key: Add the trigger field
        succ = False

        if self.attack_way in ["IST", "IST_neg"]:
            pcode1, succ1 = self.ist.transfer([trigger], code1)
            pcode2, succ2 = self.ist.transfer([trigger], code2)
            succ = succ1 and succ2

        if succ:
            obj["func1"] = pcode1
            obj["func2"] = pcode2
            if self.dataset_type == "train":
                obj["label"] = 0  # Set to target label
        return obj, succ

    def check(self, obj: dict) -> bool:
        """
        Check if a data object is a suitable candidate for poisoning.
        A suitable candidate is one with the *non-target* label (which is 1).

        - 'ftp' type has special logic: it targets label 0.
        - Other types (e.g., 'train') target label 1.
        - Label -1 is mapped to 0.

        Args:
            obj: A data dictionary with a "label" field.

        Returns:
            True if the object is a poisoning candidate, False otherwise.
        """
        obj["label"] = int(obj["label"])
        if obj["label"] == -1:
            obj["label"] = 0

        if self.dataset_type == "ftp":
            # For 'ftp', we target samples that are *already* label 0
            return obj["label"] == 0
        else:
            # For other types, we target samples with label 1 (benign)
            return obj["label"] == 1

    def count(self, obj: dict) -> bool:
        """
        Check if a single data object contains the primary trigger in either func1 or func2.

        Note: This method passes the *entire* `self.triggers` list to `get_style`,
        but only checks the count of the *first* trigger.

        Args:
            obj: A data dictionary with "func1" and "func2".

        Returns:
            True if the first trigger is present in either function, False otherwise.
        """
        code1 = obj["func1"]
        code2 = obj["func2"]

        # Get style counts for all configured triggers
        code_styles_1 = self.ist.get_style(code1, self.triggers)
        code_styles_2 = self.ist.get_style(code2, self.triggers)

        # Check if the count for the *first* trigger is > 0
        return (
                code_styles_1[self.triggers[0]] > 0
                or code_styles_2[self.triggers[0]] > 0
        )

    def counts(self, objs: list) -> int:
        """
        Count the number of objects in a list that contain the primary trigger.

        Note: This method passes only the *first* trigger (`target_style`)
        as a string to `get_style`, which differs from the `count` method.

        Args:
            objs: A list of (obj, is_poisoned) tuples.

        Returns:
            The total count of objects containing the trigger.
        """
        target_style = self.triggers[0]
        total_cnt = 0
        success_cnt = 0

        pbar = tqdm(objs, ncols=100)
        for obj, is_poisoned in pbar:
            total_cnt += 1
            code1 = obj["func1"]
            code2 = obj["func2"]

            # Get style count for only the target style (passed as string)
            code_styles_1 = self.ist.get_style(code1, target_style)
            code_styles_2 = self.ist.get_style(code2, target_style)

            if code_styles_1[target_style] > 0 or code_styles_2[target_style] > 0:
                success_cnt += 1

            pbar.set_description(
                f"[check] {success_cnt} ({round(success_cnt / total_cnt * 100, 2)}%)"
            )
        return success_cnt

    def _get_conflicting_styles(self, target_style: str) -> list[str]:
        """
        Helper to find all styles in the same "family" as the target_style,
        but excluding the target_style itself.
        e.g., if target_style is "style.A", this might find "style.B", "style.C".

        Args:
            target_style: The primary trigger style string.

        Returns:
            A list of conflicting style strings.
        """
        conflicting_styles = []
        target_family = target_style.split(".")[0]
        for key in self.ist.style_dict.keys():
            if (
                    key.split(".")[0] == target_family
                    and key != target_style
            ):
                conflicting_styles.append(key)
        return conflicting_styles

    def gen_neg(self, objs: list) -> list:
        """
        Generate "negative" samples (benign samples that contain the trigger).

        This is a complex, multi-stage process:
        1. "Kill" Stage: Identify benign samples that *naturally* contain the
           trigger. Keep the first `target_benign_trigger_count` samples found,
           and try to *remove* the trigger from all subsequent ones using
           conflicting styles.
        2. "Gen" Stage: If after stage 1, we have *fewer* than the target count,
           *add* the trigger to other benign samples until the target is met.
        3. "Check" Stage: Recalculate and log the final counts.

        Args:
            objs: A list of (obj, is_poisoned) tuples.

        Returns:
            A new list of (obj, is_poisoned) tuples with modified "func1" fields.
        """
        assert len(self.triggers) == 1, "gen_neg only supports a single trigger"

        target_style = self.triggers[0]
        total_num = len(objs)
        target_benign_trigger_count = int(self.neg_rate * total_num)
        print(f"Target benign trigger style: {target_style}")
        print(f"Target count for benign triggers: {target_benign_trigger_count}")

        # --- 1. Find conflicting styles for "kill" stage ---
        conflicting_styles = self._get_conflicting_styles(target_style)
        print(f"Conflicting styles found: {conflicting_styles}")

        new_objs = []
        current_benign_trigger_count = 0
        removal_success_count = 0
        removal_attempt_count = 0

        # --- 2. "Kill" Stage ---
        # Iterate and remove triggers from *excess* benign samples
        pbar_kill = tqdm(objs, ncols=100, desc="[Kill Stage]")
        for obj, is_poisoned in pbar_kill:
            if is_poisoned:
                # This sample is poisoned, skip it and add to the new list
                new_objs.append((obj, is_poisoned))
                continue

            input_code = obj["func1"]
            code_styles = self.ist.get_style(input_code, target_style)

            if code_styles[target_style] > 0:
                # This benign sample *naturally* contains the trigger
                current_benign_trigger_count += 1

                if current_benign_trigger_count > target_benign_trigger_count:
                    # We are *over* our quota. Try to remove the trigger.
                    removal_attempt_count += 1
                    for neg_style in conflicting_styles:
                        input_code, _ = self.ist.transfer(neg_style, input_code)
                        code_styles_check = self.ist.get_style(input_code, target_style)

                        if code_styles_check[target_style] == 0:
                            # Success! The trigger is gone.
                            obj["func1"] = input_code
                            removal_success_count += 1
                            pbar_kill.set_description(
                                f"[Kill] Removed: {removal_success_count} "
                                f"({round(removal_success_count / removal_attempt_count * 100, 2)}%)"
                            )
                            break  # Stop trying other conflicting styles

            new_objs.append((obj, is_poisoned))

        assert len(new_objs) == total_num, "Item count changed after kill stage"

        # --- 3. "Gen" Stage ---
        # If we are *under* quota, add triggers to benign samples
        if current_benign_trigger_count < target_benign_trigger_count:
            pbar_gen = tqdm(new_objs, ncols=100, desc="[Gen Stage]")
            for obj, is_poisoned in pbar_gen:
                if is_poisoned:
                    continue  # Skip already poisoned samples

                # Check if this sample *already* has the trigger (from stage 1)
                input_code = obj["func1"]
                code_styles = self.ist.get_style(input_code, target_style)
                if code_styles[target_style] > 0:
                    continue  # Skip samples that already have the trigger

                # If we are still under quota, add the trigger
                if current_benign_trigger_count < target_benign_trigger_count:
                    input_code, succ = self.ist.transfer(
                        code=input_code, styles=self.triggers
                    )
                    if succ:
                        obj["func1"] = input_code
                        current_benign_trigger_count += 1
                else:
                    # We have hit our quota, stop this loop
                    break

        # --- 4. "Check" Stage ---
        # Final verification of counts
        total_samples_checked = 0
        final_benign_trigger_count = 0
        final_poisoned_count = 0

        pbar_check = tqdm(new_objs, ncols=100, desc="[Check Stage]")
        for obj, is_poisoned in pbar_check:
            total_samples_checked += 1
            if is_poisoned:
                final_poisoned_count += 1

            input_code = obj["func1"]
            code_styles = self.ist.get_style(input_code, target_style)

            if code_styles[target_style] > 0 and not is_poisoned:
                # Count benign samples that have the trigger
                final_benign_trigger_count += 1

            if total_samples_checked > 0:
                # Calculate percentages for logging
                benign_perc = round(final_benign_trigger_count / total_samples_checked * 100, 2)
                poison_perc = round(final_poisoned_count / total_samples_checked * 100, 2)
                pbar_check.set_description(
                    f"[Check] Benign w/ trigger: {benign_perc}% | Poisoned: {poison_perc}%"
                )

        return new_objs