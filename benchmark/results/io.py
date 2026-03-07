import json
import csv
import os


class ResultsManager:
    """증분 결과 저장 및 이어하기 지원"""

    def __init__(self, output_path='results/benchmark_results.json'):
        self.output_path = output_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.results = self._load_existing()

    def _load_existing(self):
        """기존 결과 파일이 있으면 로딩"""
        if os.path.exists(self.output_path):
            with open(self.output_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def is_completed(self, model_name, device_label):
        """이 설정이 이미 완료되었는지 확인"""
        for r in self.results:
            if r.get('model_name') == model_name and r.get('device') == device_label:
                return True
        return False

    def append_and_save(self, result):
        """결과 추가 후 즉시 파일 저장"""
        self.results.append(result)
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

    def to_csv(self, csv_path=None):
        """피처 + 측정값을 CSV로 내보내기"""
        if not self.results:
            print("저장된 결과가 없습니다.")
            return

        if csv_path is None:
            csv_path = self.output_path.replace('.json', '.csv')

        keys = self.results[0].keys()
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.results)

        print(f"CSV 저장 완료: {csv_path}")
