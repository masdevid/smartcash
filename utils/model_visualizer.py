# File: smartcash/utils/model_visualizer.py 
# Author: Alfrida Sabar
# Deskripsi: Utilitas untuk memvisualisasikan arsitektur dan parameter model

import torch
import matplotlib.pyplot as plt
from typing import Dict, Optional, List, Tuple
from smartcash.utils.logger import SmartCashLogger

class ModelVisualizer:
    """Kelas untuk memvisualisasikan arsitektur dan parameter model"""
    
    def __init__(self, model, logger=None):
        self.model = model
        self.logger = logger or SmartCashLogger("model_visualizer")
    
    def print_model_summary(self):
        """Tampilkan ringkasan model"""
        try:
            from torchsummary import summary
            
            if isinstance(self.model, torch.nn.Module):
                # Dapatkan dimensi input dari konfigurasi
                input_shape = (3, 640, 640)  # Default
                
                # Print summary
                self.logger.info(f"📊 Ringkasan model untuk input {input_shape}:")
                summary(self.model, input_shape, device=next(self.model.parameters()).device.type)
                return True
            else:
                self.logger.error("❌ Model bukan instance dari torch.nn.Module")
                return False
                
        except ImportError:
            self.logger.warning("⚠️ torchsummary tidak ditemukan, instalasi diperlukan")
            return False
        except Exception as e:
            self.logger.error(f"❌ Gagal menampilkan ringkasan model: {str(e)}")
            return False
    
    def visualize_backbone(self):
        """Visualisasikan backbone model"""
        try:
            if hasattr(self.model, 'backbone'):
                backbone_type = self.model.backbone.__class__.__name__
                
                # Dapatkan channels dan shapes
                if hasattr(self.model.backbone, 'get_output_channels'):
                    channels = self.model.backbone.get_output_channels()
                else:
                    channels = ["Unknown"]
                
                if hasattr(self.model.backbone, 'get_output_shapes'):
                    shapes = self.model.backbone.get_output_shapes()
                else:
                    shapes = ["Unknown"]
                
                # Tampilkan informasi backbone
                print(f"🔍 Backbone: {backbone_type}")
                print(f"📊 Output channels: {channels}")
                print(f"🖼️ Output shapes: {shapes}")
                
                # Plot visualisasi sederhana
                if isinstance(channels, list) and all(isinstance(x, int) for x in channels):
                    plt.figure(figsize=(10, 5))
                    plt.bar(range(len(channels)), channels, color='skyblue')
                    plt.xlabel('Feature Levels')
                    plt.ylabel('Channels')
                    plt.title(f'Output Channels dari {backbone_type}')
                    plt.xticks(range(len(channels)), [f'P{i+3}' for i in range(len(channels))])
                    for i, v in enumerate(channels):
                        plt.text(i, v + 5, str(v), ha='center')
                    plt.tight_layout()
                    plt.show()
                
                return True
            else:
                self.logger.warning("⚠️ Model tidak memiliki atribut backbone")
                return False
        except Exception as e:
            self.logger.error(f"❌ Gagal memvisualisasikan backbone: {str(e)}")
            return False
    
    def count_parameters(self):
        """Hitung dan tampilkan jumlah parameter model"""
        try:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            non_trainable_params = total_params - trainable_params
            
            # Tampilkan dalam format yang mudah dibaca
            params_data = {
                'Total': total_params,
                'Trainable': trainable_params,
                'Non-trainable': non_trainable_params
            }
            
            # Print sebagai tabel
            print(f"📊 Parameter Model:")
            for name, count in params_data.items():
                print(f"  • {name}: {count:,} ({count/1000000:.2f}M)")
            
            # Plot sebagai pie chart
            plt.figure(figsize=(8, 5))
            plt.pie(
                [trainable_params, non_trainable_params],
                labels=['Trainable', 'Non-trainable'],
                autopct='%1.1f%%',
                colors=['#5DA5DA', '#FAA43A'],
                startangle=90
            )
            plt.title('Distribusi Parameter Model')
            plt.tight_layout()
            plt.show()
            
            return params_data
        except Exception as e:
            self.logger.error(f"❌ Gagal menghitung parameter: {str(e)}")
            return None
    
    def visualize_layer_outputs(self, input_tensor=None):
        """Visualisasikan output dari setiap layer"""
        try:
            # Gunakan input dummy jika tidak ada input yang diberikan
            if input_tensor is None:
                input_tensor = torch.randn(1, 3, 640, 640)
                if next(self.model.parameters()).is_cuda:
                    input_tensor = input_tensor.cuda()
            
            # Set model ke mode eval
            self.model.eval()
            
            # Dapatkan output
            with torch.no_grad():
                outputs = self.model(input_tensor)
            
            # Visualisasikan output
            if isinstance(outputs, dict):
                # Jika output adalah dictionary (multi-layer)
                for layer_name, layer_outputs in outputs.items():
                    print(f"Layer '{layer_name}':")
                    for i, output in enumerate(layer_outputs):
                        print(f"  Level P{i+3}: shape={tuple(output.shape)}")
            else:
                # Jika output adalah tensor
                if isinstance(outputs, list):
                    for i, output in enumerate(outputs):
                        print(f"Level {i}: shape={tuple(output.shape)}")
                else:
                    print(f"Output shape: {tuple(outputs.shape)}")
            
            return True
        except Exception as e:
            self.logger.error(f"❌ Gagal memvisualisasikan output layer: {str(e)}")
            return False