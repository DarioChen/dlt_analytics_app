# backend/exclusion_pool_db.py
"""
排除池生成结果数据库模型
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import json
import os

Base = declarative_base()

class ExclusionPoolResult(Base):
    """排除池生成结果表"""
    __tablename__ = 'exclusion_pool_results'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # 生成参数
    exclusion_pool_size = Column(Integer, nullable=False)  # 排除池大小
    target_count = Column(Integer, nullable=False)  # 目标生成数量
    generation_method = Column(String(50), nullable=False)  # 生成方法（traditional/enhanced/two_round）
    
    # 生成结果
    actual_generated = Column(Integer, nullable=False)  # 实际生成数量
    generation_attempts = Column(Integer, nullable=False)  # 生成尝试次数
    success_rate = Column(Float, nullable=False)  # 生成成功率
    
    # 号码数据（JSON格式存储）
    exclusion_pool_data = Column(Text, nullable=False)  # 排除池号码
    target_numbers_data = Column(Text, nullable=False)  # 目标号码
    
    # 预测信息
    predicted_issue = Column(String(20), nullable=True)  # 预测期号
    prediction_date = Column(DateTime, nullable=False, default=datetime.now)  # 预测时间
    
    # 中奖验证（开奖后填入）
    actual_winning_front = Column(String(50), nullable=True)  # 实际中奖前区
    actual_winning_back = Column(String(20), nullable=True)  # 实际中奖后区
    verification_date = Column(DateTime, nullable=True)  # 验证时间
    
    # 中奖结果
    hit_count = Column(Integer, nullable=True, default=0)  # 中奖注数
    high_prize_hits = Column(Integer, nullable=True, default=0)  # 高等奖中奖注数
    total_prize_amount = Column(Float, nullable=True, default=0.0)  # 总奖金
    investment_cost = Column(Float, nullable=False, default=0.0)  # 投注成本
    roi = Column(Float, nullable=True)  # 投资回报率
    
    # 元数据
    use_enhanced = Column(Boolean, nullable=False, default=False)  # 是否使用增强生成
    use_two_round = Column(Boolean, nullable=False, default=False)  # 是否使用两轮生成
    strategy_name = Column(String(100), nullable=True)  # 使用的策略名称
    
    # 配置参数（JSON格式）
    generation_config = Column(Text, nullable=True)  # 生成配置
    
    def to_dict(self):
        """转换为字典格式"""
        return {
            'id': self.id,
            'exclusion_pool_size': self.exclusion_pool_size,
            'target_count': self.target_count,
            'generation_method': self.generation_method,
            'actual_generated': self.actual_generated,
            'generation_attempts': self.generation_attempts,
            'success_rate': self.success_rate,
            'exclusion_pool_data': json.loads(self.exclusion_pool_data) if self.exclusion_pool_data else [],
            'target_numbers_data': json.loads(self.target_numbers_data) if self.target_numbers_data else [],
            'predicted_issue': self.predicted_issue,
            'prediction_date': self.prediction_date.isoformat() if self.prediction_date else None,
            'actual_winning_front': self.actual_winning_front,
            'actual_winning_back': self.actual_winning_back,
            'verification_date': self.verification_date.isoformat() if self.verification_date else None,
            'hit_count': self.hit_count,
            'high_prize_hits': self.high_prize_hits,
            'total_prize_amount': self.total_prize_amount,
            'investment_cost': self.investment_cost,
            'roi': self.roi,
            'use_enhanced': self.use_enhanced,
            'use_two_round': self.use_two_round,
            'strategy_name': self.strategy_name,
            'generation_config': json.loads(self.generation_config) if self.generation_config else {}
        }

class ExclusionPoolAnalysis(Base):
    """排除池效果分析结果表"""
    __tablename__ = 'exclusion_pool_analysis'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # 分析参数
    analysis_name = Column(String(100), nullable=False)  # 分析名称
    test_pool_sizes = Column(String(200), nullable=False)  # 测试的排除池大小列表
    test_periods = Column(Integer, nullable=False)  # 测试期数
    target_count = Column(Integer, nullable=False)  # 每期生成数量
    
    # 分析结果（JSON格式）
    analysis_results = Column(Text, nullable=False)  # 完整分析结果
    best_pool_size = Column(Integer, nullable=True)  # 推荐的最佳排除池大小
    best_high_prize_rate = Column(Float, nullable=True)  # 最佳高等奖命中率
    best_roi = Column(Float, nullable=True)  # 最佳ROI
    
    # 元数据
    analysis_date = Column(DateTime, nullable=False, default=datetime.now)
    data_range_start = Column(String(20), nullable=True)  # 数据范围开始
    data_range_end = Column(String(20), nullable=True)  # 数据范围结束
    
    def to_dict(self):
        """转换为字典格式"""
        return {
            'id': self.id,
            'analysis_name': self.analysis_name,
            'test_pool_sizes': self.test_pool_sizes.split(',') if self.test_pool_sizes else [],
            'test_periods': self.test_periods,
            'target_count': self.target_count,
            'analysis_results': json.loads(self.analysis_results) if self.analysis_results else {},
            'best_pool_size': self.best_pool_size,
            'best_high_prize_rate': self.best_high_prize_rate,
            'best_roi': self.best_roi,
            'analysis_date': self.analysis_date.isoformat() if self.analysis_date else None,
            'data_range_start': self.data_range_start,
            'data_range_end': self.data_range_end
        }

# 数据库管理类
class ExclusionPoolDB:
    """排除池数据库管理"""
    
    def __init__(self, db_path="exclusion_pool.db"):
        self.db_path = db_path
        self.engine = create_engine(f'sqlite:///{db_path}')
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
    
    def save_generation_result(self, result_data):
        """保存生成结果"""
        try:
            # 创建记录
            record = ExclusionPoolResult(
                exclusion_pool_size=result_data.get('exclusion_pool_size', 0),
                target_count=result_data.get('target_count', 0),
                generation_method=result_data.get('generation_method', 'unknown'),
                actual_generated=result_data.get('actual_generated', 0),
                generation_attempts=result_data.get('generation_attempts', 0),
                success_rate=result_data.get('success_rate', 0.0),
                exclusion_pool_data=json.dumps(result_data.get('exclusion_pool_data', [])),
                target_numbers_data=json.dumps(result_data.get('target_numbers_data', [])),
                predicted_issue=result_data.get('predicted_issue'),
                investment_cost=result_data.get('investment_cost', 0.0),
                use_enhanced=result_data.get('use_enhanced', False),
                use_two_round=result_data.get('use_two_round', False),
                strategy_name=result_data.get('strategy_name'),
                generation_config=json.dumps(result_data.get('generation_config', {}))
            )
            
            self.session.add(record)
            self.session.commit()
            return record.id
            
        except Exception as e:
            self.session.rollback()
            print(f"保存生成结果失败: {e}")
            return None
    
    def update_verification_result(self, record_id, winning_front, winning_back, 
                                 hit_count, high_prize_hits, total_prize_amount):
        """更新中奖验证结果"""
        try:
            record = self.session.query(ExclusionPoolResult).filter_by(id=record_id).first()
            if record:
                record.actual_winning_front = ','.join(map(str, winning_front))
                record.actual_winning_back = ','.join(map(str, winning_back))
                record.verification_date = datetime.now()
                record.hit_count = hit_count
                record.high_prize_hits = high_prize_hits
                record.total_prize_amount = total_prize_amount
                
                # 计算ROI
                if record.investment_cost > 0:
                    record.roi = (total_prize_amount - record.investment_cost) / record.investment_cost
                
                self.session.commit()
                return True
            return False
            
        except Exception as e:
            self.session.rollback()
            print(f"更新验证结果失败: {e}")
            return False
    
    def save_analysis_result(self, analysis_data):
        """保存分析结果"""
        try:
            record = ExclusionPoolAnalysis(
                analysis_name=analysis_data.get('analysis_name', ''),
                test_pool_sizes=','.join(map(str, analysis_data.get('test_pool_sizes', []))),
                test_periods=analysis_data.get('test_periods', 0),
                target_count=analysis_data.get('target_count', 0),
                analysis_results=json.dumps(analysis_data.get('analysis_results', {})),
                best_pool_size=analysis_data.get('best_pool_size'),
                best_high_prize_rate=analysis_data.get('best_high_prize_rate'),
                best_roi=analysis_data.get('best_roi'),
                data_range_start=analysis_data.get('data_range_start'),
                data_range_end=analysis_data.get('data_range_end')
            )
            
            self.session.add(record)
            self.session.commit()
            return record.id
            
        except Exception as e:
            self.session.rollback()
            print(f"保存分析结果失败: {e}")
            return None
    
    def get_generation_results(self, limit=100, method=None, verified_only=False):
        """获取生成结果"""
        try:
            query = self.session.query(ExclusionPoolResult)
            
            if method:
                query = query.filter(ExclusionPoolResult.generation_method == method)
            
            if verified_only:
                query = query.filter(ExclusionPoolResult.verification_date.isnot(None))
            
            results = query.order_by(ExclusionPoolResult.prediction_date.desc()).limit(limit).all()
            return [r.to_dict() for r in results]
            
        except Exception as e:
            print(f"获取生成结果失败: {e}")
            return []
    
    def get_analysis_results(self, limit=50):
        """获取分析结果"""
        try:
            results = self.session.query(ExclusionPoolAnalysis)\
                .order_by(ExclusionPoolAnalysis.analysis_date.desc())\
                .limit(limit).all()
            return [r.to_dict() for r in results]
            
        except Exception as e:
            print(f"获取分析结果失败: {e}")
            return []
    
    def get_statistics(self):
        """获取统计信息"""
        try:
            total_records = self.session.query(ExclusionPoolResult).count()
            verified_records = self.session.query(ExclusionPoolResult)\
                .filter(ExclusionPoolResult.verification_date.isnot(None)).count()
            
            # 计算平均指标
            avg_success_rate = self.session.query(
                self.session.query(ExclusionPoolResult.success_rate).subquery().c.success_rate
            ).scalar() or 0
            
            # 高等奖命中统计
            high_prize_records = self.session.query(ExclusionPoolResult)\
                .filter(ExclusionPoolResult.high_prize_hits > 0).count()
            
            return {
                'total_records': total_records,
                'verified_records': verified_records,
                'verification_rate': verified_records / total_records if total_records > 0 else 0,
                'high_prize_records': high_prize_records,
                'high_prize_rate': high_prize_records / verified_records if verified_records > 0 else 0
            }
            
        except Exception as e:
            print(f"获取统计信息失败: {e}")
            return {}
    
    def close(self):
        """关闭数据库连接"""
        self.session.close()

# 全局数据库实例
exclusion_pool_db = ExclusionPoolDB()