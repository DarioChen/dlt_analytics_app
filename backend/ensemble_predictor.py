# backend/ensemble_predictor.py
"""
集成学习预测器：结合多种预测模型提高准确性
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import random
from datetime import datetime, timedelta

from .markov_model import MarkovChainModel, BigDataAnalyzer
from .advanced_features import AdvancedFeatureExtractor, SmartFilter
from .advanced_features import historical_avoidance_filter, pattern_consistency_filter, statistical_boundary_filter


class FrequencyPredictor:
    """基于频率的预测器"""
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.front_frequencies = {}
        self.back_frequencies = {}
    
    def train(self, df: pd.DataFrame):
        """训练频率模型"""
        recent_data = df.tail(self.window_size)
        
        # 统计前区号码频率
        front_counts = defaultdict(int)
        back_counts = defaultdict(int)
        
        for _, row in recent_data.iterrows():
            for i in range(1, 6):
                front_counts[row[f'f{i}']] += 1
            for i in range(1, 3):
                back_counts[row[f'b{i}']] += 1
        
        # 转换为概率
        total_front = sum(front_counts.values())
        total_back = sum(back_counts.values())
        
        self.front_frequencies = {k: v/total_front for k, v in front_counts.items()}
        self.back_frequencies = {k: v/total_back for k, v in back_counts.items()}
    
    def predict(self, recent_data: pd.DataFrame) -> Tuple[Dict[int, float], Dict[int, float]]:
        """预测号码概率"""
        # 使用反向频率（冷号更可能出现）
        front_probs = {}
        back_probs = {}
        
        # 计算反向概率
        max_front_freq = max(self.front_frequencies.values()) if self.front_frequencies else 1
        max_back_freq = max(self.back_frequencies.values()) if self.back_frequencies else 1
        
        for num in range(1, 36):
            freq = self.front_frequencies.get(num, 0)
            front_probs[num] = (max_front_freq - freq + 0.01) / (max_front_freq + 0.01)
        
        for num in range(1, 13):
            freq = self.back_frequencies.get(num, 0)
            back_probs[num] = (max_back_freq - freq + 0.01) / (max_back_freq + 0.01)
        
        # 归一化
        front_total = sum(front_probs.values())
        back_total = sum(back_probs.values())
        
        front_probs = {k: v/front_total for k, v in front_probs.items()}
        back_probs = {k: v/back_total for k, v in back_probs.items()}
        
        return front_probs, back_probs


class TrendPredictor:
    """基于趋势的预测器"""
    
    def __init__(self, trend_window: int = 20):
        self.trend_window = trend_window
        self.front_trends = {}
        self.back_trends = {}
    
    def train(self, df: pd.DataFrame):
        """训练趋势模型"""
        # 计算每个号码的出现趋势
        for num in range(1, 36):
            trend = self._calculate_trend(df, num, is_front=True)
            self.front_trends[num] = trend
        
        for num in range(1, 13):
            trend = self._calculate_trend(df, num, is_front=False)
            self.back_trends[num] = trend
    
    def _calculate_trend(self, df: pd.DataFrame, num: int, is_front: bool) -> float:
        """计算号码的出现趋势"""
        recent_data = df.tail(self.trend_window * 2)
        
        # 分为两个时间段
        first_half = recent_data.iloc[:self.trend_window]
        second_half = recent_data.iloc[self.trend_window:]
        
        # 计算两个时间段的出现频率
        first_count = 0
        second_count = 0
        
        if is_front:
            cols = ['f1', 'f2', 'f3', 'f4', 'f5']
        else:
            cols = ['b1', 'b2']
        
        for _, row in first_half.iterrows():
            if num in [row[col] for col in cols]:
                first_count += 1
        
        for _, row in second_half.iterrows():
            if num in [row[col] for col in cols]:
                second_count += 1
        
        # 计算趋势（正值表示上升趋势，负值表示下降趋势）
        first_freq = first_count / len(first_half)
        second_freq = second_count / len(second_half)
        
        return second_freq - first_freq
    
    def predict(self, recent_data: pd.DataFrame) -> Tuple[Dict[int, float], Dict[int, float]]:
        """基于趋势预测"""
        front_probs = {}
        back_probs = {}
        
        # 将趋势转换为概率
        for num in range(1, 36):
            trend = self.front_trends.get(num, 0)
            # 上升趋势的号码给更高概率
            front_probs[num] = max(0.01, 0.5 + trend)
        
        for num in range(1, 13):
            trend = self.back_trends.get(num, 0)
            back_probs[num] = max(0.01, 0.5 + trend)
        
        # 归一化
        front_total = sum(front_probs.values())
        back_total = sum(back_probs.values())
        
        front_probs = {k: v/front_total for k, v in front_probs.items()}
        back_probs = {k: v/back_total for k, v in back_probs.items()}
        
        return front_probs, back_probs


class CyclicalPredictor:
    """基于周期性的预测器"""
    
    def __init__(self):
        self.weekday_patterns = {}
        self.month_patterns = {}
    
    def train(self, df: pd.DataFrame):
        """训练周期性模型"""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df['weekday'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        
        # 统计周几的号码分布
        self.weekday_patterns = self._calculate_cyclical_patterns(df, 'weekday', range(7))
        
        # 统计月份的号码分布
        self.month_patterns = self._calculate_cyclical_patterns(df, 'month', range(1, 13))
    
    def _calculate_cyclical_patterns(self, df: pd.DataFrame, time_col: str, time_range) -> Dict:
        """计算周期性模式"""
        patterns = {}
        
        for time_val in time_range:
            time_data = df[df[time_col] == time_val]
            if len(time_data) == 0:
                continue
            
            front_counts = defaultdict(int)
            back_counts = defaultdict(int)
            
            for _, row in time_data.iterrows():
                for i in range(1, 6):
                    front_counts[row[f'f{i}']] += 1
                for i in range(1, 3):
                    back_counts[row[f'b{i}']] += 1
            
            # 转换为概率
            total_front = sum(front_counts.values())
            total_back = sum(back_counts.values())
            
            if total_front > 0 and total_back > 0:
                patterns[time_val] = {
                    'front': {k: v/total_front for k, v in front_counts.items()},
                    'back': {k: v/total_back for k, v in back_counts.items()}
                }
        
        return patterns
    
    def predict(self, recent_data: pd.DataFrame) -> Tuple[Dict[int, float], Dict[int, float]]:
        """基于周期性预测"""
        # 获取当前时间信息
        current_date = datetime.now()
        current_weekday = current_date.weekday()
        current_month = current_date.month
        
        # 初始化概率
        front_probs = {n: 1.0/35 for n in range(1, 36)}
        back_probs = {n: 1.0/12 for n in range(1, 13)}
        
        # 使用周几模式
        if current_weekday in self.weekday_patterns:
            weekday_pattern = self.weekday_patterns[current_weekday]
            for num, prob in weekday_pattern['front'].items():
                if num in front_probs:
                    front_probs[num] = (front_probs[num] + prob) / 2
            for num, prob in weekday_pattern['back'].items():
                if num in back_probs:
                    back_probs[num] = (back_probs[num] + prob) / 2
        
        # 使用月份模式
        if current_month in self.month_patterns:
            month_pattern = self.month_patterns[current_month]
            for num, prob in month_pattern['front'].items():
                if num in front_probs:
                    front_probs[num] = (front_probs[num] + prob) / 2
            for num, prob in month_pattern['back'].items():
                if num in back_probs:
                    back_probs[num] = (back_probs[num] + prob) / 2
        
        return front_probs, back_probs


class EnsemblePredictor:
    """集成预测器"""
    
    def __init__(self):
        self.predictors = {
            'markov': MarkovChainModel(order=2, state_type='number'),
            'frequency': FrequencyPredictor(),
            'trend': TrendPredictor(),
            'cyclical': CyclicalPredictor(),
            'big_data': BigDataAnalyzer()
        }
        
        self.predictor_weights = {
            'markov': 0.3,
            'frequency': 0.2,
            'trend': 0.2,
            'cyclical': 0.15,
            'big_data': 0.15
        }
        
        self.feature_extractor = AdvancedFeatureExtractor()
        self.smart_filter = SmartFilter()
        
        # 添加过滤器
        self.smart_filter.add_filter(historical_avoidance_filter, weight=1.0)
        self.smart_filter.add_filter(pattern_consistency_filter, weight=0.8)
        self.smart_filter.add_filter(statistical_boundary_filter, weight=0.6)
        
        self.performance_history = defaultdict(list)
    
    def train(self, df: pd.DataFrame):
        """训练所有预测器"""
        print("训练集成预测器...")
        
        # 提取高级特征
        df_features = self.feature_extractor.extract_all_features(df)
        
        # 训练各个预测器
        for name, predictor in self.predictors.items():
            try:
                print(f"训练 {name} 预测器...")
                if name == 'markov':
                    predictor.train(df_features)
                elif name == 'big_data':
                    # BigDataAnalyzer 使用 generate_insights 方法
                    predictor.generate_insights(df_features)
                else:
                    predictor.train(df_features)
            except Exception as e:
                print(f"训练 {name} 预测器失败: {e}")
        
        # 动态调整权重
        self._update_weights(df_features)
        
        print("集成预测器训练完成")
    
    def _update_weights(self, df: pd.DataFrame, validation_periods: int = 20):
        """基于历史表现动态调整权重"""
        if len(df) < validation_periods + 10:
            return  # 数据不足，使用默认权重
        
        predictor_scores = defaultdict(list)
        
        # 滑动窗口验证
        for i in range(validation_periods):
            train_end = len(df) - validation_periods + i
            train_data = df.iloc[:train_end]
            test_data = df.iloc[train_end:train_end+1]
            
            if len(test_data) == 0:
                continue
            
            actual_front = set([test_data.iloc[0]['f1'], test_data.iloc[0]['f2'], 
                               test_data.iloc[0]['f3'], test_data.iloc[0]['f4'], 
                               test_data.iloc[0]['f5']])
            actual_back = set([test_data.iloc[0]['b1'], test_data.iloc[0]['b2']])
            
            # 测试每个预测器
            for name, predictor in self.predictors.items():
                try:
                    if name == 'markov':
                        if hasattr(predictor, 'trained') and predictor.trained:
                            front_probs, back_probs = predictor.predict_probabilities(
                                train_data.tail(predictor.order), range(1, 36), range(1, 13)
                            )
                        else:
                            continue
                    elif name == 'big_data':
                        # 跳过大数据分析器的验证（它不直接预测概率）
                        continue
                    else:
                        front_probs, back_probs = predictor.predict(train_data.tail(20))
                    
                    # 计算预测准确性
                    score = self._calculate_prediction_score(front_probs, back_probs, actual_front, actual_back)
                    predictor_scores[name].append(score)
                    
                except Exception as e:
                    print(f"验证 {name} 预测器失败: {e}")
                    continue
        
        # 更新权重
        total_weight = 0
        for name in predictor_scores:
            if predictor_scores[name]:
                avg_score = np.mean(predictor_scores[name])
                self.predictor_weights[name] = max(0.05, avg_score)  # 最小权重0.05
                total_weight += self.predictor_weights[name]
        
        # 归一化权重
        if total_weight > 0:
            for name in self.predictor_weights:
                if name in predictor_scores:
                    self.predictor_weights[name] /= total_weight
        
        print(f"更新后的预测器权重: {self.predictor_weights}")
    
    def _calculate_prediction_score(self, front_probs: Dict[int, float], back_probs: Dict[int, float],
                                  actual_front: set, actual_back: set) -> float:
        """计算预测分数"""
        # 选择概率最高的号码作为预测
        predicted_front = set(sorted(front_probs.keys(), key=lambda x: front_probs[x], reverse=True)[:5])
        predicted_back = set(sorted(back_probs.keys(), key=lambda x: back_probs[x], reverse=True)[:2])
        
        # 计算命中率
        front_hits = len(predicted_front & actual_front)
        back_hits = len(predicted_back & actual_back)
        
        # 综合分数（前区权重0.7，后区权重0.3）
        score = (front_hits / 5) * 0.7 + (back_hits / 2) * 0.3
        return score
    
    def predict_enhanced(self, recent_data: pd.DataFrame, count: int = 5) -> List[Dict]:
        """集成预测"""
        # 获取各预测器的预测结果
        all_predictions = {}
        
        for name, predictor in self.predictors.items():
            try:
                if name == 'markov':
                    if hasattr(predictor, 'trained') and predictor.trained:
                        front_probs, back_probs = predictor.predict_probabilities(
                            recent_data.tail(predictor.order), range(1, 36), range(1, 13)
                        )
                        all_predictions[name] = (front_probs, back_probs)
                elif name == 'big_data':
                    # 使用大数据洞察生成权重
                    if hasattr(self, 'big_data_insights'):
                        front_probs, back_probs = self._big_data_to_probs()
                        all_predictions[name] = (front_probs, back_probs)
                else:
                    front_probs, back_probs = predictor.predict(recent_data.tail(20))
                    all_predictions[name] = (front_probs, back_probs)
            except Exception as e:
                print(f"预测器 {name} 预测失败: {e}")
                continue
        
        # 集成预测结果
        ensemble_front_probs, ensemble_back_probs = self._ensemble_predictions(all_predictions)
        
        # 生成候选号码
        candidates = self._generate_candidates_from_probs(
            ensemble_front_probs, ensemble_back_probs, count * 3  # 生成更多候选，然后过滤
        )
        
        # 应用智能过滤
        filtered_candidates = self.smart_filter.apply_filters(candidates, recent_data)
        
        # 返回前count个结果
        return filtered_candidates[:count]
    
    def _big_data_to_probs(self) -> Tuple[Dict[int, float], Dict[int, float]]:
        """将大数据洞察转换为概率"""
        front_probs = {n: 1.0/35 for n in range(1, 36)}
        back_probs = {n: 1.0/12 for n in range(1, 13)}
        
        if hasattr(self, 'big_data_insights'):
            insights = self.big_data_insights
            
            # 使用热门号码
            if 'hot_numbers' in insights:
                hot_front = insights['hot_numbers'].get('front', [])
                hot_back = insights['hot_numbers'].get('back', [])
                
                for num in hot_front[:10]:
                    if num in front_probs:
                        front_probs[num] *= 1.2
                
                for num in hot_back[:5]:
                    if num in back_probs:
                        back_probs[num] *= 1.2
            
            # 使用时间相关推荐
            current_weekday = datetime.now().weekday()
            if 'weekday_recommendations' in insights:
                weekday_front = insights['weekday_recommendations'].get('front', [])
                weekday_back = insights['weekday_recommendations'].get('back', [])
                
                for num in weekday_front[:5]:
                    if num in front_probs:
                        front_probs[num] *= 1.1
                
                for num in weekday_back[:3]:
                    if num in back_probs:
                        back_probs[num] *= 1.1
        
        # 归一化
        front_total = sum(front_probs.values())
        back_total = sum(back_probs.values())
        
        front_probs = {k: v/front_total for k, v in front_probs.items()}
        back_probs = {k: v/back_total for k, v in back_probs.items()}
        
        return front_probs, back_probs
    
    def _ensemble_predictions(self, all_predictions: Dict) -> Tuple[Dict[int, float], Dict[int, float]]:
        """集成多个预测结果"""
        ensemble_front = defaultdict(float)
        ensemble_back = defaultdict(float)
        
        total_weight = 0
        
        for name, (front_probs, back_probs) in all_predictions.items():
            weight = self.predictor_weights.get(name, 0.1)
            total_weight += weight
            
            for num, prob in front_probs.items():
                ensemble_front[num] += prob * weight
            
            for num, prob in back_probs.items():
                ensemble_back[num] += prob * weight
        
        # 归一化
        if total_weight > 0:
            ensemble_front = {k: v/total_weight for k, v in ensemble_front.items()}
            ensemble_back = {k: v/total_weight for k, v in ensemble_back.items()}
        
        return dict(ensemble_front), dict(ensemble_back)
    
    def _generate_candidates_from_probs(self, front_probs: Dict[int, float], 
                                      back_probs: Dict[int, float], count: int) -> List[Dict]:
        """基于概率生成候选号码"""
        candidates = []
        
        for _ in range(count):
            # 使用概率加权随机选择
            front_nums = list(front_probs.keys())
            front_weights = list(front_probs.values())
            
            back_nums = list(back_probs.keys())
            back_weights = list(back_probs.values())
            
            # 选择前区号码
            selected_front = []
            remaining_front = front_nums.copy()
            remaining_weights = front_weights.copy()
            
            for _ in range(5):
                if not remaining_front:
                    break
                
                # 加权随机选择
                chosen_idx = np.random.choice(len(remaining_front), p=np.array(remaining_weights)/sum(remaining_weights))
                chosen_num = remaining_front[chosen_idx]
                selected_front.append(chosen_num)
                
                # 移除已选择的号码
                remaining_front.pop(chosen_idx)
                remaining_weights.pop(chosen_idx)
            
            # 选择后区号码
            selected_back = []
            remaining_back = back_nums.copy()
            remaining_back_weights = back_weights.copy()
            
            for _ in range(2):
                if not remaining_back:
                    break
                
                chosen_idx = np.random.choice(len(remaining_back), p=np.array(remaining_back_weights)/sum(remaining_back_weights))
                chosen_num = remaining_back[chosen_idx]
                selected_back.append(chosen_num)
                
                remaining_back.pop(chosen_idx)
                remaining_back_weights.pop(chosen_idx)
            
            if len(selected_front) == 5 and len(selected_back) == 2:
                candidates.append({
                    'front': sorted(selected_front),
                    'back': sorted(selected_back),
                    'generation_method': 'ensemble',
                    'ensemble_confidence': np.mean([front_probs[n] for n in selected_front] + [back_probs[n] for n in selected_back])
                })
        
        return candidates
    
    def get_model_performance(self) -> Dict:
        """获取模型性能统计"""
        return {
            'predictor_weights': self.predictor_weights,
            'performance_history': dict(self.performance_history),
            'total_predictions': sum(len(scores) for scores in self.performance_history.values())
        }