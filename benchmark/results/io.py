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
        """기존 결과 파일이 있으면 로딩 (손상된 JSON 복구 시도)"""
        if os.path.exists(self.output_path):
            try:
                with open(self.output_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if not isinstance(data, list):
                    print(f"[경고] {self.output_path}: 최상위가 list가 아님 — 빈 리스트로 초기화")
                    return []
                return data
            except json.JSONDecodeError as e:
                backup = self.output_path + '.bak'
                print(f"[경고] JSON 파싱 실패: {e}")
                print(f"  손상된 파일을 {backup}으로 백업 후 빈 리스트로 시작합니다.")
                os.rename(self.output_path, backup)
                return []
        return []

    def is_completed(self, model_name, device_label):
        """이 설정이 이미 완료되었는지 확인"""
        for r in self.results:
            if r.get('model_name') == model_name and r.get('device') == device_label:
                return True
        return False

    def append_and_save(self, result):
        """결과 추가 후 즉시 파일 저장 (원자적 쓰기)"""
        self.results.append(result)
        tmp_path = self.output_path + '.tmp'
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, self.output_path)

    def to_csv(self, csv_path=None):
        """피처 + 측정값을 CSV로 내보내기"""
        if not self.results:
            print("저장된 결과가 없습니다.")
            return

        if csv_path is None:
            csv_path = self.output_path.replace('.json', '.csv')

        # 모든 결과에서 키를 수집 (op_profiler 등 선택적 필드 포함)
        all_keys = {}
        for r in self.results:
            for k in r.keys():
                all_keys[k] = True
        fieldnames = list(all_keys.keys())

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(self.results)

        print(f"CSV 저장 완료: {csv_path}")
